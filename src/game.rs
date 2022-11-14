use anyhow::{bail, Ok, Result};
use ndarray::{array, Array, Array2};

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellState {
    Empty,
    O,
    X,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    O,
    X,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameState {
    InAction,
    OWin,
    XWin,
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OXGame {
    pub board: Vec<CellState>,
    pub result: GameState,
}

impl OXGame {
    pub fn new() -> Self {
        OXGame {
            board: vec![CellState::Empty; 9],
            result: GameState::InAction,
        }
    }

    pub fn step(mut self, pos: usize, player: Player) -> Result<Self> {
        if self.result == GameState::InAction {
            match self.board[pos] {
                CellState::Empty => {
                    self.put(pos, player)?;
                    self.result = self.get_game_state()?;
                }
                _ => bail!("The piece is already placed in that cell."),
            }
        } else {
            bail!("Game is over.")
        }
        Ok(self.clone())
    }

    pub fn show(&self) {
        println!(
            " {} | {} | {} ",
            self.board[0], self.board[1], self.board[2]
        );
        println!("---+---+---");
        println!(
            " {} | {} | {} ",
            self.board[3], self.board[4], self.board[5]
        );
        println!("---+---+---");
        println!(
            " {} | {} | {} ",
            self.board[6], self.board[7], self.board[8]
        );
    }

    pub fn show_w_number(&self) {
        let mut strs = Vec::new();
        for i in 0..9 {
            match self.board[i] {
                CellState::Empty => strs.push(format!("{}", i)),
                _ => strs.push(format!("{}", self.board[i])),
            };
        }

        println!(" {} | {} | {} ", strs[0], strs[1], strs[2]);
        println!("---+---+---");
        println!(" {} | {} | {} ", strs[3], strs[4], strs[5]);
        println!("---+---+---");
        println!(" {} | {} | {} ", strs[6], strs[7], strs[8]);
    }

    fn put(&mut self, pos: usize, player: Player) -> Result<()> {
        let future_cell = match player {
            Player::O => CellState::O,
            Player::X => CellState::X,
        };

        self.board[pos] = future_cell;
        Ok(())
    }

    fn get_game_state(&self) -> Result<GameState> {
        if self.is_win_with_player(Player::O)? {
            Ok(GameState::OWin)
        } else if self.is_win_with_player(Player::X)? {
            Ok(GameState::XWin)
        } else if self.board.clone().iter().any(|c| c == &CellState::Empty) {
            Ok(GameState::InAction)
        } else {
            Ok(GameState::Draw)
        }
    }

    fn is_win_with_player(&self, player: Player) -> Result<bool> {
        let player_state = match player {
            Player::O => CellState::O,
            Player::X => CellState::X,
        };
        let state_int = self
            .clone()
            .board
            .into_iter()
            .map(|s| Self::cell_state_to_int(s, player_state))
            .collect::<Vec<isize>>();

        let result = Self::is_win(state_int)?;

        Ok(result)
    }

    fn is_win(board: Vec<isize>) -> Result<bool> {
        let board_arr: Array2<isize> = Array::from_shape_vec((3, 3), board).unwrap();
        let trans = board_arr.clone().reversed_axes();

        let mask: Array2<isize> = Array::from_shape_vec((3, 1), vec![1, 1, 1]).unwrap();
        let diagonal_mask_lu_to_rd: Array2<isize> = Array2::eye(3);
        let diagonal_mask_ld_to_ru: Array2<isize> = array![[0, 0, 1], [0, 1, 0], [1, 0, 0]];

        // Check if the rows are complete.
        let masked = board_arr.dot(&mask).into_shape(3)?;
        let max_masked = masked.into_iter().max().unwrap();

        // Check if the cols are complete.
        let masked_trans = trans.dot(&mask).into_shape(3)?;
        let max_masked_trans = masked_trans.into_iter().max().unwrap();

        // Check if the diagonals are complete.
        let diagonal_masked_lu_to_rd = &board_arr * diagonal_mask_lu_to_rd;
        let diagonal_masked_ld_to_ru = &board_arr * diagonal_mask_ld_to_ru;
        let diagonal_sum_lu_to_rd: isize = diagonal_masked_lu_to_rd.into_iter().sum();
        let diagonal_sum_ld_to_ru: isize = diagonal_masked_ld_to_ru.into_iter().sum();

        if max_masked == 3 {
            Ok(true)
        } else if max_masked_trans == 3 {
            Ok(true)
        } else if diagonal_sum_lu_to_rd == 3 {
            Ok(true)
        } else if diagonal_sum_ld_to_ru == 3 {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn cell_state_to_int(state: CellState, judge_state: CellState) -> isize {
        if state == judge_state {
            1
        } else {
            0
        }
    }
}

impl fmt::Display for CellState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match self {
            CellState::Empty => " ",
            CellState::O => "O",
            CellState::X => "X",
        };
        write!(f, "{}", str)
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match self {
            Player::O => "O",
            Player::X => "X",
        };
        write!(f, "{}", str)
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match self {
            GameState::Draw => "draw",
            GameState::InAction => "in action",
            GameState::OWin => "O win",
            GameState::XWin => "X win",
        };
        write!(f, "{}", str)
    }
}
