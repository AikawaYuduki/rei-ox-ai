extern crate tensorflow;

use anyhow::Result;

use rei_ox_ai::game::*;
use rei_ox_ai::model::*;

use rand::seq::SliceRandom;
use std::io;

fn main() -> Result<()> {
    let mut _game = OXGame::new();
    let _agent = Agent::new(Player::O)?;
    let _ = _agent.action(&_game.board, &_game);

    let mut game = OXGame::new();

    // choose player turn
    let mut choices = vec![Player::O, Player::X];
    let mut rng = rand::thread_rng();
    choices.shuffle(&mut rng);
    let player = choices[0];
    let rival = choices[1];

    let agent = Agent::new(rival)?;

    // start game
    println!("Your mark is {}", player);
    let mut now_turn = Player::O;
    loop {
        match game.result {
            GameState::InAction => {}
            GameState::Draw => {
                println!("Draw!!!");
                break;
            }
            GameState::OWin => {
                match player {
                    Player::O => println!("You Win!!!"),
                    Player::X => println!("Rei Win!!!"),
                }
                game.show();
                break;
            }
            GameState::XWin => {
                match player {
                    Player::X => println!("You Win!!!"),
                    Player::O => println!("Rei Win!!!"),
                }
                game.show();
                break;
            }
        };
        if player == now_turn {
            println!("");
            game.show_w_number();
            println!("Your turn.");
            println!("Select the number of the cell you wish to place.\n");
            game = loop {
                let mut inp = String::new();
                io::stdin().read_line(&mut inp).expect("Input Error!");
                let input_number = match inp.trim().parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => 9,
                };

                if input_number < 9 {
                    match game.board[input_number] {
                        CellState::Empty => break game.step(input_number, player)?,
                        _ => println!("The selected cell is already marked."),
                    };
                } else {
                    println!("Incorrect input.");
                    println!("Enter a number between 0 and 8.");
                };
            }
        } else {
            println!("");
            game.show_w_number();
            println!("Rei turn.\n");
            game = agent.action(&game.board, &game)?;
        }
        now_turn = match now_turn {
            Player::O => Player::X,
            Player::X => Player::O,
        };
    }

    Ok(())
}
