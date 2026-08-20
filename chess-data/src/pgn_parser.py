import chess.pgn
import io
from typing import Dict, Any

def parse_game_pgn(pgn_string: str) -> Dict[str, Any]:
    """
    Parses a PGN string and extracts relevant metadata and the full move text.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {}

    headers = game.headers

    # Extract required fields, handling missing keys safely
    data = {
        "game_id": headers.get("Link", "").split("/")[-1] if "Link" in headers else "",
        "date": headers.get("UTCDate", headers.get("Date", "")),
        "white": headers.get("White", ""),
        "black": headers.get("Black", ""),
        "white_rating": headers.get("WhiteElo", ""),
        "black_rating": headers.get("BlackElo", ""),
        "result": headers.get("Result", ""),
        "time_control": headers.get("TimeControl", ""),
        "rated": headers.get("Event", "").lower().find("rated") != -1 or "Rated" in headers.get("Event", ""),
        "eco": headers.get("ECO", ""),
        "opening": headers.get("ECOUrl", "").split("/")[-1].replace("-", " ") if "ECOUrl" in headers else "",
        "termination": headers.get("Termination", ""),
        "event": headers.get("Event", ""),
        "pgn": pgn_string
    }

    # As fallback for 'rated' since chess.com PGN often just says "Live Chess" in Event
    # We can refine this later or rely on the JSON 'rated' field during extraction, 
    # but since this parses PGN only, we'll keep it simple.
    
    # We might not need to strictly determine rated from Event if Event just says "Live Chess"
    # Actually, the JSON object has a "rated": true boolean. We will pass the JSON object to processor.

    return data

def extract_moves_only(pgn_string: str) -> str:
    """
    Extracts only the moves from a PGN string, discarding headers.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return ""
    
    exporter = chess.pgn.StringExporter(headers=False, variations=True, comments=True)
    return game.accept(exporter)
