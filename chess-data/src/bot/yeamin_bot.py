import chess
from src.engine.stockfish_engine import StockfishEngine
from src.inference.yeamin_style_model import YeaminStyleModel
from src.chess_features.extractor import extract_position_features, extract_candidate_features
from src.preprocessing.pipeline import build_inference_dataframe

class YeaminBot:
    def __init__(self, engine_path="bin/stockfish.exe", model_path="data/phase4/models/yeamin_style_model.pkl"):
        self.engine = StockfishEngine(path=engine_path)
        self.model = YeaminStyleModel(model_path=model_path)
        
    def predict_move(self, fen, style_strength=0.75, max_cpl=100, candidate_count=5):
        pos_features = extract_position_features(fen)
        
        sf_candidates = self.engine.get_top_moves(fen, multipv=candidate_count)
        
        if not sf_candidates:
            return None # Game over
            
        for cand in sf_candidates:
            cand_feats = extract_candidate_features(fen, cand['candidate_move'])
            cand.update(cand_feats)
            
        df = build_inference_dataframe(pos_features, sf_candidates)
        
        style_probs = self.model.rank_candidates(df)
        
        best_candidate = None
        best_score = -float('inf')
        results = []
        
        for i, cand in enumerate(sf_candidates):
            style_prob = style_probs[i]
            engine_eval = cand['candidate_eval']
            eval_drop = cand['eval_drop']
            
            engine_prob = self.engine.get_win_probability(engine_eval)
            
            # Combine engine strength and personal style
            final_score = (1.0 - style_strength) * engine_prob + style_strength * style_prob
            
            # Safety threshold: reject if eval_drop is worse than max_cpl
            is_rejected = False
            if eval_drop < -max_cpl:
                is_rejected = True
                final_score = -1000.0
                
            # Verify legality
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(cand['candidate_move'])
            if move_obj not in board.legal_moves:
                is_rejected = True
                final_score = -2000.0
                
            res = {
                'move': cand['candidate_move'],
                'style_probability': round(float(style_prob), 4),
                'stockfish_rank': cand['stockfish_rank'],
                'stockfish_eval': int(engine_eval),
                'eval_drop': int(eval_drop),
                'engine_win_prob': round(float(engine_prob), 4),
                'final_score': round(float(final_score), 4),
                'is_rejected': is_rejected
            }
            results.append(res)
            
            if final_score > best_score and not is_rejected:
                best_score = final_score
                best_candidate = res
                
        # Fallback to Stockfish best move if ALL candidates are rejected
        if best_candidate is None:
            best_candidate = results[0]
            
        # Generate explanation
        reason = ""
        if best_candidate['stockfish_rank'] == 1:
            reason = "The engine's first choice perfectly aligns with your historical preferences or safety margins."
        else:
            reason = "Your historical model preferred this type of candidate over the engine's first choice."
            
        return {
            'fen': fen,
            'move': best_candidate['move'],
            'style_strength': style_strength,
            'confidence': best_candidate['final_score'],
            'stockfish_best': results[0]['move'],
            'selected_rank': best_candidate['stockfish_rank'],
            'explanation': reason,
            'candidates': results
        }
