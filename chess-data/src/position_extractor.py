import chess
import chess.pgn
from typing import List, Dict, Any, Optional, Tuple
from src.validators import validate_position

def extract_positions_from_game(game: chess.pgn.Game, target_username: str, start_pos_id: int) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Extracts positions where the target_username is to move.
    Returns: (list_of_positions, next_pos_id, parse_error_count)
    """
    positions = []
    headers = game.headers
    
    white_player = headers.get("White", "")
    black_player = headers.get("Black", "")
    
    # Check if target player is in the game
    is_white = target_username.lower() == white_player.lower()
    is_black = target_username.lower() == black_player.lower()
    
    if not is_white and not is_black:
        return positions, start_pos_id, 0
        
    game_id = headers.get("Link", "").split("/")[-1] if "Link" in headers else ""
    date = headers.get("UTCDate", headers.get("Date", ""))
    result = headers.get("Result", "")
    time_control = headers.get("TimeControl", "")
    eco = headers.get("ECO", "")
    opening = headers.get("ECOUrl", "").split("/")[-1].replace("-", " ") if "ECOUrl" in headers else ""
    
    your_color = "White" if is_white else "Black"
    your_rating = headers.get("WhiteElo", "") if is_white else headers.get("BlackElo", "")
    opp_rating = headers.get("BlackElo", "") if is_white else headers.get("WhiteElo", "")
    
    board = game.board()
    pos_id = start_pos_id
    errors = 0
    
    # Iterate through the moves
    for node in game.mainline():
        move = node.move
        turn = board.turn # True for White, False for Black
        
        # Check if it's our turn BEFORE the move is made
        is_our_turn = (turn == chess.WHITE and is_white) or (turn == chess.BLACK and is_black)
        
        if is_our_turn:
            # Validate
            if validate_position(board, move):
                pos_data = {
                    "position_id": pos_id,
                    "game_id": game_id,
                    "move_number": board.fullmove_number,
                    "ply": board.ply(),
                    "fen": board.fen(),
                    "side_to_move": "White" if turn == chess.WHITE else "Black",
                    "your_move": move.uci(),
                    "your_move_san": board.san(move),
                    "your_color": your_color,
                    "result": result,
                    "time_control": time_control,
                    "your_rating": your_rating,
                    "opponent_rating": opp_rating,
                    "date": date,
                    "eco": eco,
                    "opening": opening
                }
                positions.append(pos_data)
                pos_id += 1
            else:
                errors += 1
                
        # Make the move on the board
        board.push(move)
        
    return positions, pos_id, errors
