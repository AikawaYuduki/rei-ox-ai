import argparse
import random
from collections import deque

import numpy as np
import oxgame
from fn_framework import Experience, FNAgent, Observer, Trainer
from PIL import Image
from tensorflow import keras as K


class DeepQNetworkAgent(FNAgent):
    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(
            K.layers.Dense(
                81, kernel_initializer=normal, activation="relu", input_shape=(9,)
            )
        )
        model.add(K.layers.Dense(243, kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(81, kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences, gamma):
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss

    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())


class OXObserver(Observer):
    def __init__(self, env: oxgame.OXGame):
        super().__init__(env)
        self.n_step = 0
        self.transform_np_func = np.vectorize(self._transform)

    @property
    def observation_space(self):
        return self._env.board

    @property
    def player(self):
        if self.n_step % 2 == 0:
            return "O"
        else:
            return "X"

    @property
    def rival(self):
        if self.n_step % 2 == 0:
            return "X"
        else:
            return "O"

    @staticmethod
    def _transform(x):
        if x == 2.0:
            return -1.0
        else:
            return x

    def transform(self, state):
        # player -> 1, rival -> -1
        if self.player == "O":
            return self.transform_np_func(state)
        else:
            return self.transform_np_func(state) * -1

    def reset(self):
        self._env.reset()
        return self.transform(self._env.board)

    def render(self):
        out = [
            f" {self._env.board[0]} | {self._env.board[1]} | {self._env.board[2]} ",
            f"---+---+---",
            f" {self._env.board[3]} | {self._env.board[4]} | {self._env.board[5]} ",
            f"---+---+---",
            f" {self._env.board[6]} | {self._env.board[7]} | {self._env.board[8]} \n\n\n",
        ]
        out_text = "\n".join(out)
        with open(".\\py\\logs\\out\\render.log", "a") as f:
            f.write(out_text)

    def step(self, action):
        try:
            self._env.step(action, self.player)
            n_state = self.transform(self._env.board)
            if self._env.result == "InAction":
                done = False
                reward = 0
            elif self._env.result == f"{self.player}Win":
                done = True
                reward = 2
            elif self._env.result == f"{self.rival}Win":
                done = True
                reward = -1
            elif self._env.result == "Draw":
                done = True
                reward = 1
            else:
                raise Exception("something is wrong.")
        except oxgame.CollisionException:
            n_state = self.transform(self._env.board)
            done = True
            reward = -1

        if done:
            self.n_step = 0
        else:
            self.n_step += 1

        info = self._env.result

        return n_state, reward, done, info


class DeepQNetworkTrainer(Trainer):
    def __init__(
        self,
        buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        initial_epsilon=0.5,
        final_epsilon=1e-3,
        learning_rate=1e-3,
        teacher_update_freq=3,
        report_interval=10,
        log_dir="",
        file_name="",
    ):
        super().__init__(buffer_size, batch_size, gamma, report_interval, log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.teacher_update_freq = teacher_update_freq
        self.loss = 0
        self.training_episode = 0
        self._max_reward = [-10, -10]

    def train(
        self,
        env,
        episode_count=1200,
        initial_count=200,
        render=False,
        observe_interval=100,
    ):
        actions = list(range(env.action_space.n))
        agent_O = DeepQNetworkAgent(1.0, actions)
        agent_X = DeepQNetworkAgent(1.0, actions)

        self.training_episode = [episode_count, episode_count]

        self.train_loop(
            env,
            [agent_O, agent_X],
            episode_count,
            initial_count,
            render,
            observe_interval,
        )
        return agent_O

    def train_loop(
        self,
        env,
        agents,
        episode=200,
        initial_count=-1,
        render=False,
        observe_interval=0,
    ):
        self.experiences = [
            deque(maxlen=self.buffer_size),
            deque(maxlen=self.buffer_size),
        ]
        self.training = [False, False]
        self.training_count = [0, 0]
        self.reward_log = [[], []]
        frames = [[], []]

        for i in range(episode):
            init_turn = i % 2
            s = env.reset()
            done = False
            step_count = 0
            step_count_turns = [0, 0]
            self.episode_begin(i, agents[init_turn])
            while not done:
                turn = (init_turn + step_count) % 2
                if render:
                    env.render()
                if (
                    self.training[turn]
                    and observe_interval > 0
                    and (
                        self.training_count[turn] == 1
                        or self.training_count[turn] % observe_interval == 0
                    )
                ):
                    frames[turn].append(s)

                a = agents[turn].policy(s)
                n_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences[turn].append(e)
                if info == "Draw":
                    rival_s = self.experiences[1 - turn][-1].s
                    rival_a = self.experiences[1 - turn][-1].a
                    rival_n_s = self.experiences[1 - turn][-1].n_s
                    rival_d = self.experiences[1 - turn][-1].d
                    rival_e = Experience(rival_s, rival_a, reward, rival_n_s, rival_d)
                    self.experiences[1 - turn].pop()
                    self.experiences[1 - turn].append(rival_e)
                elif done and info.endswith("Win"):
                    rival_s = self.experiences[1 - turn][-1].s
                    rival_a = self.experiences[1 - turn][-1].a
                    rival_r = reward * -1
                    rival_n_s = self.experiences[1 - turn][-1].n_s
                    rival_d = self.experiences[1 - turn][-1].d
                    rival_e = Experience(rival_s, rival_a, rival_r, rival_n_s, rival_d)
                    self.experiences[1 - turn].pop()
                    self.experiences[1 - turn].append(rival_e)
                if (
                    not self.training[turn]
                    and len(self.experiences[turn]) == self.buffer_size
                ):
                    self.begin_train(i, agents[turn], turn)
                    self.training[turn] = True

                self.step(i, step_count, agents[turn], e, turn)

                s = n_state * -1
                step_count += 1
                step_count_turns[turn] += 1
            else:
                self.episode_end(i, step_count_turns[turn], agents[turn], turn)
                self.episode_end(
                    i, step_count_turns[1 - turn], agents[1 - turn], 1 - turn
                )

                if not self.training[turn] and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agents[turn], turn)
                    self.training[turn] = True

                if self.training[turn]:
                    if len(frames[turn]) > 0:
                        # self.logger.write_image(self.training_count, frames)
                        frames[turn] = []
                    self.training_count[turn] += 1

    def episode_begin(self, episode, agent):
        self.loss = [0, 0]

    def begin_train(self, episode, agent, turn):
        optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)
        agent.initialize(self.experiences[turn], optimizer)
        agent.epsilon = self.initial_epsilon
        self.training_episode[turn] -= episode

    def step(self, episode, step_count, agent, experience, turn):
        if self.training[turn]:
            batch = random.sample(self.experiences[turn], self.batch_size)
            self.loss[turn] += agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent, turn):
        reward = sum([e.r for e in self.get_recent(step_count, turn)])
        self.loss[turn] = self.loss[turn] / step_count
        self.reward_log[turn].append(reward)
        if self.training[turn]:
            if episode % 100 == 0:
                self._max_reward[turn] = -10
            if reward >= self._max_reward[turn]:
                if turn == 0:
                    dev_episode = (episode // 100) * 100
                    dev_file_name = self.file_name.split(".")
                    agent.save(
                        self.logger.path_of(
                            f"{dev_file_name[0]}_{dev_episode}.{dev_file_name[1]}"
                        )
                    )
                self._max_reward[turn] = reward
            if self.is_event(self.training_count[turn], self.teacher_update_freq):
                agent.update_teacher()

            diff = self.initial_epsilon - self.final_epsilon
            decay = diff / self.training_episode[turn]
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval) and turn == 0:
            recent_rewards = self.reward_log[turn][-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode)

    def get_recent(self, count, turn):
        recent = range(len(self.experiences[turn]) - count, len(self.experiences[turn]))
        return [self.experiences[turn][i] for i in recent]


def main(play):
    file_name = "dqn_agent.h5"
    trainer = DeepQNetworkTrainer(file_name=file_name)
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = DeepQNetworkAgent

    env = oxgame.OXGame()
    obs = OXObserver(env)
    trainer.learning_rate = 1e-4

    if play:
        agent = agent_class.load(obs, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, episode_count=200000)

    trainer.logger.plot("reward", trainer.reward_log[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")

    args = parser.parse_args()
    main(args.play)
