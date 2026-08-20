import chess
import chess.engine
import json
import os

class StockfishEngine:
    def __init__(self, path="bin/stockfish.exe", depth=18, threads=2):
        if not os.path.exists(path):
            try:
                with open("config.json") as f:
                    cfg = json.load(f)
                    path = cfg.get("stockfish_path", path)
            except:
                pass
                
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": 128})
        self.depth = depth
        
    def set_depth(self, depth):
        self.depth = depth
        
    def analyze_position(self, fen):
        board = chess.Board(fen)
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return info["score"].white().score(mate_score=10000)
        
    def get_best_move(self, fen):
        board = chess.Board(fen)
        res = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return res.move.uci()
        
    def get_top_moves(self, fen, multipv=5):
        board = chess.Board(fen)
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=multipv)
        
        candidates = []
        for i, info in enumerate(infos):
            score_pov = info["score"].pov(board.turn)
            score = score_pov.score(mate_score=10000)
            is_mate = 1 if score_pov.is_mate() else 0
            
            if "pv" not in info or not info["pv"]:
                continue
                
            candidates.append({
                'candidate_move': info["pv"][0].uci(),
                'stockfish_rank': i + 1,
                'candidate_eval': score if score is not None else 0,
                'is_mate': is_mate
            })
            
        if candidates:
            r1_score = candidates[0]['candidate_eval']
            for cand in candidates:
                cand['eval_drop'] = cand['candidate_eval'] - r1_score
                
        return candidates
        
    def get_win_probability(self, centipawns):
        if centipawns is None:
            return 0.5
        cp = max(-10000, min(10000, centipawns))
        return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))
        
    def __del__(self):
        try:
            self.engine.quit()
        except:
            pass
