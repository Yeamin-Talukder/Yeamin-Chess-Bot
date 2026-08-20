import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_charts(df: pd.DataFrame, output_dir: str):
    """
    Generate all required visualizations and save to output_dir/charts.
    """
    print("Generating charts...")
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    sns.set_theme(style="darkgrid")
    
    # Pre-filter for valid CPL to avoid extreme outliers messing up plots
    valid_cpl = df[df['centipawn_loss'] <= 1000]
    
    # 1. CPL Distribution
    plt.figure(figsize=(10,6))
    sns.histplot(valid_cpl['centipawn_loss'], bins=50, kde=True, color='#2ECC71')
    plt.title('Centipawn Loss Distribution (capped at 1000)')
    plt.xlabel('Centipawn Loss')
    plt.savefig(os.path.join(charts_dir, "1_cpl_distribution.png"), bbox_inches='tight')
    plt.close()
    
    # 2. CPL Boxplot by Game Phase
    if 'game_phase' in df.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x='game_phase', y='centipawn_loss', data=valid_cpl, showfliers=False)
        plt.title('CPL by Game Phase')
        plt.savefig(os.path.join(charts_dir, "2_cpl_by_phase.png"), bbox_inches='tight')
        plt.close()
        
    # 3. White vs Black Average CPL
    plt.figure(figsize=(6,6))
    sns.barplot(x='your_color', y='centipawn_loss', data=valid_cpl, estimator=np.mean)
    plt.title('Average CPL: White vs Black')
    plt.savefig(os.path.join(charts_dir, "3_white_vs_black.png"), bbox_inches='tight')
    plt.close()
    
    # 4. Engine Agreement (Top 1, Top 3, Top 5)
    rates = [df['is_engine_best'].mean()*100, df['is_top3'].mean()*100, df['is_top5'].mean()*100]
    labels = ['Top 1', 'Top 3', 'Top 5']
    plt.figure(figsize=(8,6))
    sns.barplot(x=labels, y=rates, palette="viridis")
    plt.title('Engine Agreement Rates (%)')
    plt.ylabel('Percentage')
    plt.savefig(os.path.join(charts_dir, "4_engine_agreement.png"), bbox_inches='tight')
    plt.close()
    
    # 5. CPL by Evaluation Category
    if 'eval_category' in df.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(x='eval_category', y='centipawn_loss', data=valid_cpl, 
                    order=['Winning', 'Advantage', 'Equal', 'Disadvantage', 'Losing'])
        plt.title('Average CPL under Pressure')
        plt.savefig(os.path.join(charts_dir, "5_cpl_by_eval.png"), bbox_inches='tight')
        plt.close()
        
    # 6. PCA Cluster Plot (if clusters exist)
    cluster_file = os.path.join(output_dir, "features", "game_clusters.csv")
    if os.path.exists(cluster_file):
        clusters = pd.read_csv(cluster_file)
        if 'pca1' in clusters.columns and 'pca2' in clusters.columns:
            plt.figure(figsize=(10,8))
            sns.scatterplot(x='pca1', y='pca2', hue='cluster', palette='tab10', data=clusters, alpha=0.6)
            plt.title('Style Clusters (PCA projection of games)')
            plt.savefig(os.path.join(charts_dir, "6_pca_clusters.png"), bbox_inches='tight')
            plt.close()
            
    print("Charts generated successfully.")
