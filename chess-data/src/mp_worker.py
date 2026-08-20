import chess.engine
from src.stockfish_analyzer import StockfishAnalyzer

# Global analyzer instance for the worker process
_analyzer = None

def init_worker(stockfish_path, threads, hash_mb):
    global _analyzer
    _analyzer = StockfishAnalyzer(stockfish_path, threads, hash_mb)

def worker_job(data):
    # data: (fen, your_move_uci, your_color, depth, multipv)
    global _analyzer
    fen, move, color, depth, multipv = data
    return _analyzer.analyze_position(fen, move, color, depth, multipv)
