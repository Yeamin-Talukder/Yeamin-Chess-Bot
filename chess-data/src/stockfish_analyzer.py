import chess
import chess.engine
import os
import sys

class StockfishAnalyzer:
    def __init__(self, executable_path: str, threads: int = 4, hash_mb: int = 512):
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Stockfish executable not found at: {executable_path}")
            
        self.engine = chess.engine.SimpleEngine.popen_uci(executable_path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        
    def analyze_position(self, fen: str, your_move_uci: str, your_color: str, depth: int = 10, multipv: int = 5) -> dict:
        """
        Analyzes a position. Normalizes evaluations so positive is good for `your_color`.
        """
        board = chess.Board(fen)
        
        # Analyze top N lines
        limit = chess.engine.Limit(depth=depth)
        info = self.engine.analyse(board, limit, multipv=multipv)
        
        # In case info is just one dict (when multipv=1 or engine doesn't support it)
        if isinstance(info, dict):
            info = [info]
            
        # Ensure it's sorted by score descending (engine usually does this, but to be safe)
        info.sort(key=lambda x: x["score"].pov(board.turn).score(mate_score=10000), reverse=True)
        
        result = {}
        
        # Extract top multipv moves
        for i, line in enumerate(info):
            rank = i + 1
            if rank <= 5:
                # Get the first move of the PV
                pv = line.get("pv", [])
                move = pv[0].uci() if pv else ""
                
                # Evaluation from board's perspective (turn to move)
                # But we want it from `your_color` perspective
                score = line["score"].pov(chess.WHITE if your_color == "White" else chess.BLACK)
                eval_cp = score.score(mate_score=10000)
                
                result[f"stockfish_rank_{rank}_move"] = move
                result[f"stockfish_rank_{rank}_eval_cp"] = eval_cp
                
                if rank == 1:
                    result["stockfish_best_move"] = move
                    result["stockfish_best_move_san"] = board.san(pv[0]) if pv else ""
                    result["stockfish_best_eval_cp"] = eval_cp
        
        # Now find evaluation of your actual move
        your_move_eval_cp = None
        your_move_rank = ">5"
        
        for i, line in enumerate(info):
            pv = line.get("pv", [])
            if pv and pv[0].uci() == your_move_uci:
                score = line["score"].pov(chess.WHITE if your_color == "White" else chess.BLACK)
                your_move_eval_cp = score.score(mate_score=10000)
                your_move_rank = str(i + 1)
                break
                
        # If your move is outside the multipv, we must explicitly evaluate it
        if your_move_eval_cp is None:
            try:
                your_chess_move = chess.Move.from_uci(your_move_uci)
                if your_chess_move in board.legal_moves:
                    # Create a root move limit to analyze ONLY this move
                    root_limit = chess.engine.Limit(depth=depth)
                    specific_info = self.engine.analyse(board, root_limit, root_moves=[your_chess_move])
                    
                    if isinstance(specific_info, list):
                        specific_info = specific_info[0]
                        
                    score = specific_info["score"].pov(chess.WHITE if your_color == "White" else chess.BLACK)
                    your_move_eval_cp = score.score(mate_score=10000)
                else:
                    your_move_eval_cp = 0 # Fallback for illegal (shouldn't happen)
            except Exception:
                your_move_eval_cp = 0
                
        result["your_move_eval_cp"] = your_move_eval_cp
        result["your_move_rank"] = your_move_rank
        
        # Calculate centipawn loss (best eval - your eval)
        best_eval = result.get("stockfish_best_eval_cp", 0)
        cpl = best_eval - your_move_eval_cp
        # CPL should conceptually be positive since best is higher than yours, but in mate situations it might jump.
        # We cap CPL to be at least 0 to avoid negative loss.
        result["centipawn_loss"] = max(0, cpl)
        
        result["stockfish_depth"] = depth
        result["stockfish_multipv"] = multipv
        
        return result

    def close(self):
        self.engine.quit()
