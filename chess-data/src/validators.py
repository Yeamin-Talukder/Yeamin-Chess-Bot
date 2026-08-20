import chess

def validate_position(board: chess.Board, move: chess.Move) -> bool:
    """
    Validates that the given board state is legal and the move is pseudo-legal
    (which implies legal if the board is valid in most normal game progressions).
    We check full legality to be safe.
    """
    if not board.is_valid():
        return False
        
    if move not in board.legal_moves:
        return False
        
    return True
