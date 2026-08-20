import json
import os

def generate_reports(best_config, res_df, latency_stats, model_size, engine_size, output_dir):
    
    profile = {
        "bot_name": "Yeamin Bot",
        "model": {
            "algorithm": "HistGradientBoosting",
            "version": "1.0"
        },
        "performance": {
            "imitation_top1": best_config['exact_imitation'],
            "average_cpl": best_config['avg_cpl_drop'],
            "median_cpl": best_config['median_cpl_drop']
        },
        "style": {
            "style_win_rate": best_config['style_win_rate'],
            "style_loss_rate": best_config['style_loss_rate'],
            "style_net": best_config['style_net']
        },
        "deployment": {
            "model_size_mb": round(model_size / (1024*1024), 2),
            "engine_size_mb": round(engine_size / (1024*1024), 2),
            "average_latency_ms": latency_stats['avg'],
            "p95_latency_ms": latency_stats['p95']
        },
        "recommended": {
            "style_strength": best_config['style_strength'],
            "max_cpl": 100,
            "stockfish_depth": 18
        }
    }
    
    with open(f"{output_dir}/summaries/bot_profile.json", "w") as f:
        json.dump(profile, f, indent=4)
        
    with open(f"{output_dir}/config/bot_config.json", "w") as f:
        json.dump(profile['recommended'], f, indent=4)
        
    html = f"""
    <html>
    <head><style>body {{ font-family: sans-serif; background: #1a1a1a; color: #ddd; }}
    h1, h2 {{ color: #2ECC71; }} 
    .card {{ background: #2c3e50; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border: 1px solid #444; text-align: left; }}
    th {{ background: #2ECC71; color: #1a1a1a; }}
    </style></head>
    <body>
    <div style="max-width: 900px; margin: auto; padding: 20px;">
    <h1>Phase 5: Playable Bot Deployment</h1>
    
    <div class="card">
        <h2>Executive Summary</h2>
        <p>The Yeamin Bot architecture is complete. Using a Style Strength of <b>{best_config['style_strength']}</b>, the bot achieves a Style Net of <b>{best_config['style_net']*100:.2f}%</b> (meaning it successfully predicts your deviations from Stockfish much more often than it makes mistakes compared to it).</p>
        <p>The average CPL sacrifice for this personalization is only <b>{best_config['avg_cpl_drop']:.2f} centipawns</b>.</p>
        <p>Overall Imitation Rate: <b>{best_config['exact_imitation']*100:.2f}%</b>.</p>
    </div>
    
    <div class="card">
        <h2>Sweep Results</h2>
        <img src="../charts/style_strength_vs_net.png" style="width:100%; max-width:600px;"><br><br>
        <img src="../charts/style_strength_vs_cpl.png" style="width:100%; max-width:600px;">
    </div>
    
    <div class="card">
        <h2>Feature Importance</h2>
        <img src="../../feature_importance/feature_importance.png" style="width:100%; max-width:600px;">
        <p>This illustrates the features the model relies on most heavily. High values indicate the model uses these heavily to decide whether to override Stockfish.</p>
    </div>
    
    <div class="card">
        <h2>Deployment Specs</h2>
        <ul>
            <li>Model Size: {model_size / (1024*1024):.2f} MB</li>
            <li>Engine Size: {engine_size / (1024*1024):.2f} MB</li>
            <li>Inference Latency: {latency_stats['avg']:.2f} ms (P95: {latency_stats['p95']:.2f} ms)</li>
        </ul>
        <p>The inference layer is fully isolated and ready for Phase 6 (Android/Web Deployment).</p>
    </div>
    
    </div></body></html>
    """
    
    with open(f"{output_dir}/reports/phase5_bot_report.html", "w") as f:
        f.write(html)
        
    print("Reports generated successfully.")
