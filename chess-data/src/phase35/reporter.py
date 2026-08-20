import json
import os

def generate_report(stats: dict, output_dir: str):
    """
    Generates the HTML report and JSON summary.
    """
    print("Generating HTML Report and JSON profile...")
    
    # 1. Save JSON
    json_path = os.path.join(output_dir, "summaries", "personal_style_profile.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    # 2. Key Findings Generation
    cpl_diff = stats['color']['white']['average_cpl'] - stats['color']['black']['average_cpl']
    color_pref = "White" if cpl_diff < 0 else "Black"
    
    findings = [
        f"Your overall Centipawn Loss is {stats['performance']['average_cpl']:.1f}, matching Stockfish's top move {stats['performance']['top1_percentage']:.1f}% of the time.",
        f"You play slightly better with {color_pref}, maintaining a {abs(cpl_diff):.1f} lower average CPL.",
        f"In the opening, your top-1 agreement is {stats['game_phase'].get('opening', {}).get('top1_percentage', 0):.1f}%."
    ]
    
    # 3. HTML Generation
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Personal Chess Style Profile</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; }}
            .findings {{ background: #e8f4f8; padding: 15px; border-left: 5px solid #3498db; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Personal Chess Style Analysis</h1>
            
            <h2>Key Findings</h2>
            <div class="findings">
                <ul>
                    {''.join(f'<li>{f}</li>' for f in findings)}
                </ul>
            </div>
            
            <h2>Overall Performance</h2>
            <p><strong>Average CPL:</strong> {stats['performance']['average_cpl']:.1f}</p>
            <p><strong>Top-1 Engine Agreement:</strong> {stats['performance']['top1_percentage']:.1f}%</p>
            <p><strong>Top-3 Engine Agreement:</strong> {stats['performance']['top3_percentage']:.1f}%</p>
            
            <h2>Charts</h2>
            <h3>Centipawn Loss Distribution</h3>
            <img src="../charts/1_cpl_distribution.png" alt="CPL Distribution">
            
            <h3>Game Phase</h3>
            <img src="../charts/2_cpl_by_phase.png" alt="CPL by Phase">
            
            <h3>Engine Agreement</h3>
            <img src="../charts/4_engine_agreement.png" alt="Engine Agreement">
            
            <h3>Pressure Analysis</h3>
            <img src="../charts/5_cpl_by_eval.png" alt="Pressure">
            
            <h2>Phase 4 Recommendation</h2>
            <p>
            Based on the analysis, a direct classification model predicting the exact UCI string (e.g. "e2e4") has an extremely high dimensional space (~1800+ possible moves). 
            Given your Top-3 agreement rate is <strong>{stats['performance']['top3_percentage']:.1f}%</strong>, we strongly recommend building a <strong>candidate-ranking model</strong> for Phase 4. 
            The ML model should evaluate the Top-5 Stockfish candidate moves and predict which of those 5 candidates you are most likely to choose, based on these extracted style features.
            </p>
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, "reports", "personal_chess_style_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Report saved to {html_path}")
