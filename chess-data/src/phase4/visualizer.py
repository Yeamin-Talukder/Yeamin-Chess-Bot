import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.inspection import permutation_importance

def generate_charts(results: list, best_model, X_test, y_test, df_test, num_cols, cat_cols, output_dir: str):
    print("Generating evaluation charts...")
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # 1. Model Comparison (Top-1, Top-3, Top-5)
    res_df = pd.DataFrame(results)
    melted = res_df.melt(id_vars=['model'], value_vars=['top1', 'top3', 'top5'], var_name='Metric', value_name='Accuracy')
    plt.figure(figsize=(10,6))
    sns.barplot(x='model', y='Accuracy', hue='Metric', data=melted, palette='viridis')
    plt.title('Model Comparison: Top-K Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(charts_dir, "1_model_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Style Imitation Breakdown (A, B, C, D) for the Best Model
    best_res = res_df[res_df['model'] == 'HistGradientBoosting'].iloc[0]
    cats = [best_res['cat_A'], best_res['cat_B'], best_res['cat_C'], best_res['cat_D']]
    labels = ['A: Model==Actual\\nSF==Actual', 'B: Model==Actual\\nSF!=Actual (Style Win)', 'C: Model!=Actual\\nSF==Actual', 'D: Model!=Actual\\nSF!=Actual']
    plt.figure(figsize=(8,8))
    plt.pie(cats, labels=labels, autopct='%1.1f%%', colors=['#2ECC71', '#3498db', '#f1c40f', '#e74c3c'])
    plt.title('Style Imitation Matrix (Best Model)')
    plt.savefig(os.path.join(charts_dir, "5_style_imitation.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy by Game Phase (using df_test)
    if 'game_phase' in df_test.columns:
        phases = df_test['game_phase'].unique()
        phase_accs = []
        for p in phases:
            subset = df_test[df_test['game_phase'] == p]
            # Since pred_rank is already calculated in evaluator, we can just compute it if we pass it back.
            # But here we just compute conditional on top-1
            if 'pred_rank' in subset.columns:
                acc = (subset[subset['label']==1]['pred_rank'] == 1).mean() * 100
                phase_accs.append({'Phase': p, 'Top-1 Accuracy': acc})
                
        if phase_accs:
            pdf = pd.DataFrame(phase_accs)
            plt.figure(figsize=(6,5))
            sns.barplot(x='Phase', y='Top-1 Accuracy', data=pdf, palette='mako')
            plt.title('Accuracy by Game Phase')
            plt.savefig(os.path.join(charts_dir, "8_accuracy_phase.png"), bbox_inches='tight')
            plt.close()
            
    # 4. Accuracy by Color
    if 'your_color' in df_test.columns:
        colors = df_test['your_color'].unique()
        color_accs = []
        for c in colors:
            subset = df_test[df_test['your_color'] == c]
            if 'pred_rank' in subset.columns:
                acc = (subset[subset['label']==1]['pred_rank'] == 1).mean() * 100
                color_accs.append({'Color': c, 'Top-1 Accuracy': acc})
        if color_accs:
            cdf = pd.DataFrame(color_accs)
            plt.figure(figsize=(5,5))
            sns.barplot(x='Color', y='Top-1 Accuracy', data=cdf, palette='coolwarm')
            plt.title('Accuracy by Color')
            plt.savefig(os.path.join(charts_dir, "10_accuracy_color.png"), bbox_inches='tight')
            plt.close()

    # 5. Feature Importance (Permutation)
    print("Calculating feature importance (sample)...")
    # Sample 5000 rows to speed up permutation importance
    if len(X_test) > 5000:
        sample_idx = np.random.choice(len(X_test), 5000, replace=False)
        X_sample = X_test.iloc[sample_idx]
        y_sample = y_test.iloc[sample_idx]
    else:
        X_sample = X_test
        y_sample = y_test
        
    try:
        result = permutation_importance(best_model, X_sample, y_sample, n_repeats=3, random_state=42, n_jobs=-1)
        importances = pd.Series(result.importances_mean, index=num_cols + cat_cols)
        importances = importances.sort_values(ascending=False).head(15)
        
        plt.figure(figsize=(10,8))
        sns.barplot(x=importances.values, y=importances.index, palette='rocket')
        plt.title('Top 15 Feature Importances (Permutation)')
        plt.savefig(os.path.join(charts_dir, "7_feature_importance.png"), bbox_inches='tight')
        plt.close()
        
        # Save features to list for reporter
        with open(os.path.join(output_dir, "experiments", "top_features.txt"), "w") as f:
            for feat in importances.index[:5]:
                f.write(f"{feat}\\n")
                
    except Exception as e:
        print(f"Feature importance failed: {e}")
