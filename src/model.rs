use anyhow::Result;
use std::fs::File;
use std::io::Read;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

use ndarray::{arr1, Array1};

use crate::game::OXGame;
use crate::game::{CellState, Player};

#[derive(Debug)]
pub struct Agent {
    observer: Observer,
    session: Session,
    graph: Graph,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observer {
    pub board: Array1<f32>,
    pub player: Player,
}

impl Agent {
    pub fn new(player: Player) -> Result<Self> {
        let filename = "model/frozen_model/dqn.pb";
        let mut graph = Graph::new();
        let mut proto = Vec::new();
        File::open(filename)?.read_to_end(&mut proto)?;
        graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
        let session = Session::new(&SessionOptions::new(), &graph)?;

        Ok(Agent {
            observer: Observer::new(player),
            session,
            graph,
        })
    }

    pub fn predict(&self, vis_board: &Vec<CellState>) -> Result<Vec<usize>> {
        let board: Array1<f32> = self.observer.transform(vis_board).board;
        let mut x = Tensor::new(&[9, 9]);
        for (i, v) in board.into_iter().enumerate() {
            x[i] = v as f32;
        }
        // Run the graph.
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.operation_by_name_required("x")?, 0, &x);
        let z = args.request_fetch(&self.graph.operation_by_name_required("Identity")?, 0);
        self.session.run(&mut args)?;

        // Check our results.
        let z_res: Tensor<f32> = args.fetch(z)?;
        let mut z_array: [f32; 9] = [0.0; 9];
        for i in 0..9 {
            z_array[i] = z_res[i];
        }
        let z_argsort = Agent::argsort(&z_array);

        Ok(z_argsort)
    }

    pub fn action(&self, vis_board: &Vec<CellState>, game: &OXGame) -> Result<OXGame> {
        let action_candidate = self.predict(vis_board)?;
        let mut can_actions = Vec::new();

        for i in action_candidate {
            if game.board[i] == CellState::Empty {
                can_actions.push(i);
            }
        }

        let new_game = game.clone().step(can_actions[0], self.observer.player)?;

        Ok(new_game)
    }

    fn argsort<T: PartialOrd>(v: &[T]) -> Vec<usize> {
        let mut idx = (0..v.len()).collect::<Vec<_>>();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let rev_idx: Vec<usize> = idx.into_iter().rev().collect();
        rev_idx
    }
}

impl Observer {
    pub fn new(player: Player) -> Self {
        Observer {
            board: arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            player,
        }
    }

    pub fn transform(&self, vis_board: &Vec<CellState>) -> Self {
        let my_cell = match self.player {
            Player::O => CellState::O,
            Player::X => CellState::X,
        };
        let transformed: Vec<f32> = vis_board
            .iter()
            .map(|c| Observer::transform_cell(c, &my_cell))
            .collect();

        let transformed_ndarray: Array1<f32> = Array1::from_vec(transformed);

        Observer {
            board: transformed_ndarray,
            player: self.player,
        }
    }

    fn transform_cell(cell: &CellState, my_cell: &CellState) -> f32 {
        if my_cell == &CellState::O {
            match cell {
                CellState::O => 1.0,
                CellState::X => -1.0,
                CellState::Empty => 0.0,
            }
        } else {
            match cell {
                CellState::X => 1.0,
                CellState::O => -1.0,
                CellState::Empty => 0.0,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::game;

    use super::*;

    #[test]
    fn transform_o() -> Result<()> {
        let mut game = game::OXGame::new();
        game = game.step(0, Player::O)?;
        game = game.step(2, Player::X)?;
        let mut obs = Observer::new(Player::O);
        obs = obs.transform(&game.board);
        assert_eq!(
            obs.board,
            arr1(&[1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        Ok(())
    }
    #[test]
    fn transform_x() -> Result<()> {
        let mut game = game::OXGame::new();
        game = game.step(0, Player::O)?;
        game = game.step(2, Player::X)?;
        let mut obs = Observer::new(Player::X);
        obs = obs.transform(&game.board);
        assert_eq!(
            obs.board,
            arr1(&[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        Ok(())
    }
}
