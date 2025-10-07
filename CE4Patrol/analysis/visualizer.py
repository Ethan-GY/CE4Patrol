import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_df):
    """生成并保存多种可视化图表"""
    # 1. 雷达图：对比不同配置在各项指标上的平均表现
    avg_scores = results_df.groupby('config')[['crs', 'accuracy', 'logic_similarity', 'action_reliability']].mean()
    
    labels = avg_scores.columns
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, row in avg_scores.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=i)
        ax.fill(angles, values, alpha=0.25)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('平均性能雷达图')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig('results/radar_chart.png')
    plt.close()
    
    # 2. 热力图：展示不同异常类别在Full_CE4Patrol配置下的性能
    full_ce_df = results_df[results_df['config'] == 'CE4Patrol_Full']
    heatmap_data = full_ce_df.groupby('category')[['crs', 'accuracy', 'logic_similarity']].mean()
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f")
    plt.title('各异常类别在CE4Patrol下的性能热力图')
    plt.tight_layout()
    plt.savefig('results/heatmap_by_category.png')
    plt.close()
