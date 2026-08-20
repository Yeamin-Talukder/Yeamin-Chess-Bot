import argparse
from src.bot.yeamin_bot import YeaminBot

def main():
    parser = argparse.ArgumentParser(description="Yeamin Chess Bot CLI")
    parser.add_argument("--fen", type=str, required=True, help="FEN string of the position")
    parser.add_argument("--style", type=float, default=0.75, help="Style strength (0.0 to 1.0)")
    parser.add_argument("--max_cpl", type=int, default=100, help="Maximum allowed CPL drop")
    parser.add_argument("--engine", type=str, default="bin/stockfish.exe", help="Path to Stockfish")
    
    args = parser.parse_args()
    
    print("Loading Yeamin Bot...")
    bot = YeaminBot(engine_path=args.engine)
    
    print(f"\nAnalyzing Position:")
    print(f"FEN: {args.fen}")
    print(f"Style Strength: {args.style}")
    print(f"Max CPL Drop: {args.max_cpl}\n")
    
    result = bot.predict_move(args.fen, style_strength=args.style, max_cpl=args.max_cpl)
    
    if result is None:
        print("No valid moves found (game over).")
        return
        
    print(f"Stockfish Best:\n    {result['stockfish_best']}")
    print(f"\nYeamin Prediction:\n    {result['move']}")
    print(f"\nStyle Strength:\n    {result['style_strength']}")
    
    print(f"\nReason:\n    {result['explanation']}\n")
    
    print("Candidates:")
    for i, cand in enumerate(result['candidates']):
        print(f"\n{i+1}. {cand['move']}")
        print(f"   Style: {cand['style_probability']:.2f}")
        print(f"   Engine Win Prob: {cand['engine_win_prob']:.2f}")
        print(f"   Stockfish Rank: {cand['stockfish_rank']}")
        print(f"   Final Score: {cand['final_score']:.2f}")
        if cand['is_rejected']:
            print("   [REJECTED: Exceeded Safety Threshold or Illegal]")
            
if __name__ == "__main__":
    main()
