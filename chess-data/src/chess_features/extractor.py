import chess

def extract_position_features(fen):
    """Extracts base position features from a FEN string."""
    board = chess.Board(fen)
    
    # Material counts
    w_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    b_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    w_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
    b_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    w_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    b_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    w_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
    b_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    w_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    b_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    w_material = w_pawns*1 + w_knights*3 + w_bishops*3 + w_rooks*5 + w_queens*9
    b_material = b_pawns*1 + b_knights*3 + b_bishops*3 + b_rooks*5 + b_queens*9
    material_balance = w_material - b_material
    
    legal_move_count = board.legal_moves.count()
    is_check = int(board.is_check())
    can_castle_w = int(board.has_castling_rights(chess.WHITE))
    can_castle_b = int(board.has_castling_rights(chess.BLACK))
    
    total_material = w_material + b_material
    if total_material >= 60:
        game_phase = 'Opening'
    elif total_material >= 30:
        game_phase = 'Middlegame'
    else:
        game_phase = 'Endgame'
        
    side_to_move = 'White' if board.turn == chess.WHITE else 'Black'
    move_number = board.fullmove_number
        
    return {
        'move_number': move_number,
        'w_pawns': w_pawns, 'b_pawns': b_pawns,
        'w_knights': w_knights, 'b_knights': b_knights,
        'w_bishops': w_bishops, 'b_bishops': b_bishops,
        'w_rooks': w_rooks, 'b_rooks': b_rooks,
        'w_queens': w_queens, 'b_queens': b_queens,
        'w_material': w_material, 'b_material': b_material,
        'material_balance': material_balance,
        'legal_move_count': legal_move_count,
        'is_check': is_check,
        'can_castle_w': can_castle_w, 'can_castle_b': can_castle_b,
        'game_phase': game_phase,
        'side_to_move': side_to_move
    }

def extract_candidate_features(fen, candidate_move_uci):
    """Extracts candidate move properties for a specific move on a FEN."""
    board = chess.Board(fen)
    
    try:
        move = chess.Move.from_uci(candidate_move_uci)
        if move not in board.legal_moves:
            # Try to handle promotions if UCI is missing it
            if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                if chess.square_rank(move.to_square) == 0 or chess.square_rank(move.to_square) == 7:
                    move = chess.Move.from_uci(candidate_move_uci + 'q')
            if move not in board.legal_moves:
                raise ValueError("Illegal move")
                
        is_capture = int(board.is_capture(move))
        is_castling = int(board.is_castling(move))
        is_promotion = 1 if move.promotion else 0
        
        piece = board.piece_at(move.from_square)
        is_pawn_move = 1 if piece and piece.piece_type == chess.PAWN else 0
        moving_piece = piece.piece_type if piece else 0
        
        board.push(move)
        cand_is_check = int(board.is_check())
        
        return {
            'cand_is_capture': is_capture,
            'cand_is_castling': is_castling,
            'cand_is_promotion': is_promotion,
            'cand_is_check': cand_is_check,
            'cand_is_pawn_move': is_pawn_move,
            'cand_moving_piece': moving_piece
        }
    except Exception:
        return {
            'cand_is_capture': 0, 'cand_is_castling': 0, 'cand_is_promotion': 0, 
            'cand_is_check': 0, 'cand_is_pawn_move': 0, 'cand_moving_piece': 0
        }
