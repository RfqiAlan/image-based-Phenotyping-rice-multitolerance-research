#!/usr/bin/env python3
"""
================================================================================
Sequential Drought-Salinity Stress Classification in Rice using Image-Based Phenotyping
Authors: Rifqi Alan Maulana et al.
Benchmark and Feature Ablation Pipeline
================================================================================
Reproduces:
1. Table 5: Multiclass Ordinal MTS Benchmark (25 Validation Folds, Repeated Stratified Nested CV)
2. Table 6: Feature Ablation Study across Feature Subsets (Gradient Boosting Machine)
3. Granular Per-Class Report (Class 1, 3, 5) for Random Forest
4. Publication Figures: Confusion Matrix and Feature Importance Rankings
"""

import os
import sys
import shutil
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.data_loader import load_and_preprocess_data
from src.models import get_model_suite
from src.evaluation import (
    run_repeated_nested_cv,
    generate_class_breakdown_table,
    plot_confusion_matrix_publication,
    plot_feature_importance_publication,
    plot_ablation_figure_publication
)

def main():
    print("=" * 85)
    print("IMAGE-BASED PHENOTYPING & LEAKAGE-SAFE MACHINE LEARNING PIPELINE")
    print("Screening Sequential Drought and Salinity Tolerance in Rice (Oryza sativa L.)")
    print("=" * 85)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # 1. Load and align phenotypic datasets
    print("\n[Step 1] Loading and Aligning Phenotypic Datasets...")
    data = load_and_preprocess_data(data_dir="data")
    X = data['X']
    X_all = data['X_all']
    y_mts = data['y_mts']
    y_binary = data['y_binary']
    
    print(f"  ✓ Total Plants Verified: {len(X)}")
    print(f"  ✓ Static Features Extracted: {X.shape[1]}")
    print(f"  ✓ Full Feature Set (Static + Dynamic Deltas): {X_all.shape[1]}")
    print("  ✓ Target Class Breakdown (MTS Classes 1, 3, 5):")
    for cls_val in [1, 3, 5]:
        count = sum(y_mts == cls_val)
        pct = (count / len(y_mts)) * 100
        print(f"      - Class {cls_val}: {count} plants ({pct:.1f}%)")
        
    # 2. Benchmark all models (Repeated Stratified 5-Fold Nested CV, 25 Folds)
    print("\n[Step 2] Running Repeated Stratified 5-Fold Nested CV (25 Validation Folds)...")
    models = get_model_suite(random_state=42)
    summary_ordinal, raw_eval, consensus_oof = run_repeated_nested_cv(
        models, X_all, y_mts, n_splits=5, n_repeats=5, random_state=42
    )
    
    print("\n" + "=" * 85)
    print("TABLE 5: MULTICLASS ORDINAL MTS CLASSIFICATION BENCHMARK (25 VALIDATION FOLDS)")
    print("=" * 85)
    print(summary_ordinal)
    print("=" * 85)
    
    table5_path = os.path.join("results", "table5_model_benchmark.csv")
    summary_ordinal.to_csv(table5_path)
    print(f"[Results] Saved Table 5 to: {table5_path}")
    
    # 3. Detailed per-class breakdown (Class 1, 3, 5) for Random Forest
    print("\n[Step 3] Generating Granular Per-Class Report for Random Forest...")
    rf_oof = consensus_oof['Random Forest']
    table_per_class = generate_class_breakdown_table(y_mts, rf_oof, model_name="Random Forest")
    
    print("\n" + "-" * 75)
    print("DETAILED PER-CLASS CLASSIFICATION REPORT FOR RANDOM FOREST (N=148)")
    print("-" * 75)
    print(table_per_class.to_string(index=False))
    print("-" * 75)
    
    class_report_path = os.path.join("results", "table6_class_breakdown_rf.csv")
    table_per_class.to_csv(class_report_path, index=False)
    print(f"[Results] Saved Per-Class Report to: {class_report_path}")
    
    # 4. Feature ablation benchmark across phenotypic subsets
    print("\n[Step 4] Running Feature Ablation Study across Modalities (GBM, 25 Folds)...")
    base_traits = ['Area', 'Perim', 'Widht', 'Height', 'Major', 'Minor', 'Circle', 'Solidity', 'GLI', 'NGRDI']
    morph_cols = [c for c in X.columns if any(t in c for t in ['Area', 'Perim', 'Widht', 'Height', 'Major', 'Minor', 'Circle', 'Solidity'])]
    color_cols = [c for c in X.columns if any(t in c for t in ['GLI', 'NGRDI'])]
    delta_cols = [c for c in X_all.columns if 'Delta_' in c or 'Ratio_' in c]
    
    ablation_subsets = {
        'Colorimetry Only (Static)': color_cols,
        'Dynamic Delta (Inter-stage) Only': delta_cols,
        'Static Both (Morph + Color)': morph_cols + color_cols,
        'Morphology Only (Static)': morph_cols,
        'All Combined (Static + Dynamic Delta)': list(X_all.columns)
    }
    
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    ablation_records = []
    gbm_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    for sub_name, sub_cols in ablation_subsets.items():
        X_sub = X_all[sub_cols]
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_sub, y_binary)):
            X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
            y_train, y_test = y_binary[train_idx], y_binary[test_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            
            sw = compute_sample_weight('balanced', y_train)
            gbm_model.fit(X_tr_s, y_train, sample_weight=sw)
            y_pred = gbm_model.predict(X_te_s)
            
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            
            ablation_records.append({
                'Feature Subset Configuration': sub_name,
                'Num_Features': len(sub_cols),
                'Macro_F1': macro_f1,
                'Balanced_Acc': bal_acc,
                'Accuracy': acc
            })
            
    ablation_summary = pd.DataFrame(ablation_records).groupby(['Feature Subset Configuration', 'Num_Features']).agg({
        'Macro_F1': ['mean', 'std'],
        'Balanced_Acc': ['mean', 'std'],
        'Accuracy': ['mean', 'std']
    }).round(3)
    
    print("\n" + "=" * 85)
    print("TABLE 6: FEATURE ABLATION STUDY RESULTS (GBM on Binary MTS, 25 Folds)")
    print("=" * 85)
    print(ablation_summary)
    print("=" * 85)
    
    ablation_path = os.path.join("results", "table_ablation_study.csv")
    ablation_summary.to_csv(ablation_path)
    print(f"[Results] Saved Table 6 Ablation Study to: {ablation_path}")
    
    # 5. Generate publication figures
    print("\n[Step 5] Generating Publication-Quality Visualizations...")
    cm_path = os.path.join("results", "confusion_matrix_rf.png")
    plot_confusion_matrix_publication(y_mts, rf_oof, output_path=cm_path, model_name="Random Forest")
    
    # Feature importances using full training set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    rf_full = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    rf_full.fit(X_scaled, y_mts)
    
    imp_path = os.path.join("results", "feature_importance_rf.png")
    plot_feature_importance_publication(list(X_all.columns), rf_full.feature_importances_, output_path=imp_path, model_name="Random Forest")
    
    # Generate formal Figure 10 ablation benchmark
    ablation_fig_path = os.path.join("results", "Figure10_Feature_Ablation_Benchmark.png")
    plot_ablation_figure_publication(output_path=ablation_fig_path)
    if os.path.exists("figures"):
        shutil.copy(ablation_fig_path, os.path.join("figures", "Figure10_Feature_Ablation_Benchmark.png"))
    
    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully.")
    print("All output tables and figures have been exported to 'results/'.")
    print("=" * 80)

if __name__ == "__main__":
    main()
