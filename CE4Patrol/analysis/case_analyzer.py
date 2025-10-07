def analyze_failures(results_df, crs_threshold=0.4):
    """筛选并分类低分样本"""
    low_crs_samples = results_df[results_df['crs'] < crs_threshold].copy()
    
    def classify_error(row):
        if row['accuracy'] == 0:
            return "分类错误 (A=0)"
        if row['logic_similarity'] < 0.5:
            return "逻辑错误 (L<0.5)"
        if row['action_reliability'] < 0.5:
            return "行动建议错误 (R<0.5)"
        return "置信度过低 (C-low)"
        
    if not low_crs_samples.empty:
        low_crs_samples['error_type'] = low_crs_samples.apply(classify_error, axis=1)
        
        print("\n--- 低分案例分析报告 ---")
        print(f"筛选标准: CRS < {crs_threshold}")
        print(f"总计 {len(low_crs_samples)} 个低分案例")
        
        error_summary = low_crs_samples['error_type'].value_counts()
        print("\n错误类型分布:")
        print(error_summary)
        
        print("\n错误案例详情 (每个类型最多展示2例):")
        for error_type, group in low_crs_samples.groupby('error_type'):
            print(f"\n--- 类型: {error_type} ---")
            for _, row in group.head(2).iterrows():
                print(f"  Case ID: {row['case_id']}, Config: {row['config']}, CRS: {row['crs']:.2f}")
                print(f"  Model Reason: {row['model_output']['reason']}")
    else:
        print("未发现CRS低于阈值的失败案例。")

# 在 main.py 中调用
# from ce4patrol.analysis.case_analyzer import analyze_failures
# analyze_failures(results_df)
