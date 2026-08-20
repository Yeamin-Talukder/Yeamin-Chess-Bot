import { Chess } from 'chess.js';

export interface PositionFeatures {
    move_number: number;
    w_pawns: number;
    b_pawns: number;
    w_knights: number;
    b_knights: number;
    w_bishops: number;
    b_bishops: number;
    w_rooks: number;
    b_rooks: number;
    w_queens: number;
    b_queens: number;
    w_material: number;
    b_material: number;
    material_balance: number;
    legal_move_count: number;
    is_check: number;
    can_castle_w: number;
    can_castle_b: number;
    game_phase: string;
    side_to_move: string;
    
    // Fallbacks (from pipeline.py)
    time_control?: string;
    eco?: string;
    opening?: string;
}

export interface CandidateFeatures {
    cand_is_capture: number;
    cand_is_castling: number;
    cand_is_promotion: number;
    cand_is_check: number;
    cand_is_pawn_move: number;
    cand_moving_piece: number; // Will map to piece type integer
}

export class FeatureExtractor {
    static PIECE_VALUES: Record<string, number> = {
        'p': 1,
        'n': 3,
        'b': 3,
        'r': 5,
        'q': 9,
        'k': 0
    };

    static PIECE_TYPES: Record<string, number> = {
        'p': 1,
        'n': 2,
        'b': 3,
        'r': 4,
        'q': 5,
        'k': 6
    };

    static extractPositionFeatures(fen: string): PositionFeatures {
        const chess = new Chess(fen);
        const board = chess.board();

        let counts: Record<string, number> = {
            'w_p': 0, 'b_p': 0,
            'w_n': 0, 'b_n': 0,
            'w_b': 0, 'b_b': 0,
            'w_r': 0, 'b_r': 0,
            'w_q': 0, 'b_q': 0,
        };

        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const piece = board[i][j];
                if (piece && piece.type !== 'k') {
                    counts[`${piece.color}_${piece.type}`]++;
                }
            }
        }

        const w_material = counts['w_p']*1 + counts['w_n']*3 + counts['w_b']*3 + counts['w_r']*5 + counts['w_q']*9;
        const b_material = counts['b_p']*1 + counts['b_n']*3 + counts['b_b']*3 + counts['b_r']*5 + counts['b_q']*9;
        const material_balance = w_material - b_material;

        const total_material = w_material + b_material;
        let game_phase = 'Endgame';
        if (total_material >= 60) {
            game_phase = 'Opening';
        } else if (total_material >= 30) {
            game_phase = 'Middlegame';
        }

        const side_to_move = chess.turn() === 'w' ? 'White' : 'Black';
        
        // has_castling_rights logic. In chess.js, castling rights are in the FEN.
        // FEN format: 1st piece placement, 2nd turn, 3rd castling rights.
        const castlingRights = fen.split(' ')[2];
        const can_castle_w = (castlingRights.includes('K') || castlingRights.includes('Q')) ? 1 : 0;
        const can_castle_b = (castlingRights.includes('k') || castlingRights.includes('q')) ? 1 : 0;

        return {
            move_number: chess.moveNumber(),
            w_pawns: counts['w_p'], b_pawns: counts['b_p'],
            w_knights: counts['w_n'], b_knights: counts['b_n'],
            w_bishops: counts['w_b'], b_bishops: counts['b_b'],
            w_rooks: counts['w_r'], b_rooks: counts['b_r'],
            w_queens: counts['w_q'], b_queens: counts['b_q'],
            w_material,
            b_material,
            material_balance,
            legal_move_count: chess.moves().length,
            is_check: chess.inCheck() ? 1 : 0,
            can_castle_w,
            can_castle_b,
            game_phase,
            side_to_move,
            time_control: 'Unknown',
            eco: 'Unknown',
            opening: 'Unknown'
        };
    }

    static extractCandidateFeatures(fen: string, candidateMoveUci: string): CandidateFeatures {
        const chess = new Chess(fen);
        
        // Convert UCI to from/to/promotion
        const from = candidateMoveUci.substring(0, 2);
        const to = candidateMoveUci.substring(2, 4);
        let promotion = candidateMoveUci.length > 4 ? candidateMoveUci.substring(4, 5) : undefined;
        
        // python-chess sometimes omits promotion for queen.
        const pieceAtFrom = chess.get(from as any);
        if (pieceAtFrom && pieceAtFrom.type === 'p' && !promotion) {
            const toRank = to.charAt(1);
            if (toRank === '1' || toRank === '8') {
                promotion = 'q';
            }
        }

        try {
            const move = chess.move({ from, to, promotion });
            
            return {
                cand_is_capture: (move.flags.includes('c') || move.flags.includes('e')) ? 1 : 0,
                cand_is_castling: (move.flags.includes('k') || move.flags.includes('q')) ? 1 : 0,
                cand_is_promotion: move.promotion ? 1 : 0,
                cand_is_check: chess.inCheck() ? 1 : 0,
                cand_is_pawn_move: pieceAtFrom?.type === 'p' ? 1 : 0,
                cand_moving_piece: pieceAtFrom ? this.PIECE_TYPES[pieceAtFrom.type] : 0
            };
        } catch (e) {
            return {
                cand_is_capture: 0,
                cand_is_castling: 0,
                cand_is_promotion: 0,
                cand_is_check: 0,
                cand_is_pawn_move: 0,
                cand_moving_piece: 0
            };
        }
    }
}
