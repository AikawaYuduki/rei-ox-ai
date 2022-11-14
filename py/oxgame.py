import numpy as np


class InternalException(Exception):
    pass


class CollisionException(Exception):
    pass


class GameOverException(Exception):
    pass


class ActionSpace:
    def __init__(self, actions):
        self.actions = actions
        self.n = len(self.actions)


class OXGame:
    board: np.ndarray
    result: str
    action_space: ActionSpace

    def __init__(self):
        self.board = np.zeros(9)
        self.result = "InAction"
        self.action_space = ActionSpace(list(range(9)))

    def reset(self):
        self.board = np.zeros(9)
        self.result = "InAction"

    def step(self, pos: int, player: str) -> None:
        if self.result == "InAction":
            if self.board[pos] == 0.0:
                self.put(pos, player)
                self.result = self.get_game_state()
            else:
                raise CollisionException("The piece is already placed in that cell.")
        else:
            raise GameOverException("Game is over.")

    def put(self, pos: int, player: str) -> None:
        # player <-> board
        #    "O"        1.
        #    "X"        2.
        if player == "O":
            self.board[pos] = 1.0
        elif player == "X":
            self.board[pos] = 2.0
        else:
            raise InternalException('Player must be "O" or "X".')

    def get_game_state(self) -> str:
        if self.is_win_with_player("O"):
            return "OWin"
        elif self.is_win_with_player("X"):
            return "XWin"
        elif np.any(self.board == 0.0):
            return "InAction"
        else:
            return "Draw"

    def is_win_with_player(self, player: str) -> bool:
        if player == "O":
            player_state = 1.0
        elif player == "X":
            player_state = 2.0
        else:
            raise InternalException('Player must be "O" or "X".')

        state_int_func = np.vectorize(lambda x: 1.0 if x == player_state else 0.0)

        state_int = state_int_func(self.board)

        result = self.is_win(state_int)

        return result

    @staticmethod
    def is_win(board: np.ndarray) -> bool:
        rs_board = board.reshape([3, 3])
        t_board = rs_board.T

        mask = np.ones((3, 1))
        diagonal_mask_lu_to_rd = np.eye(3)
        diagonal_mask_ld_to_ru = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )

        # Check if the rows are complete.
        masked = np.dot(rs_board, mask)
        max_masked = masked.max()

        # Check if the cols are complete.
        masked_trans = np.dot(t_board, mask)
        max_masked_trans = masked_trans.max()

        # Check if the diagonals are complete.
        diagonal_masked_lu_to_rd = rs_board * diagonal_mask_lu_to_rd
        diagonal_masked_ld_to_ru = rs_board * diagonal_mask_ld_to_ru

        diagonal_sum_lu_to_rd = diagonal_masked_lu_to_rd.sum()
        diagonal_sum_ld_to_ru = diagonal_masked_ld_to_ru.sum()

        if max_masked == 3.0:
            return True
        elif max_masked_trans == 3.0:
            return True
        elif diagonal_sum_lu_to_rd == 3.0:
            return True
        elif diagonal_sum_ld_to_ru == 3.0:
            return True
        else:
            return False


if __name__ == "__main__":
    game = OXGame()
    game.step(0, "O")
    game.step(1, "X")
    game.step(3, "O")
    game.step(6, "X")
    game.step(2, "O")
    game.step(4, "X")
    game.step(7, "O")
    game.step(5, "X")
    game.step(8, "O")
