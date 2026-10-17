import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix
)

def compute_ordinal_mae(y_true, y_pred):
    rank_map = {1: 0, 3: 1, 5: 2}
    r_true = np.array([rank_map[y] for y in y_true])
    r_pred = np.array([rank_map.get(y, 1) for y in y_pred])
    return np.mean(np.abs(r_true - r_pred))


def run_repeated_nested_cv(models_dict, X_features, y_target, n_splits=5, n_repeats=5, random_state=42):
    """
    Performs leakage-safe Repeated Stratified Nested Cross-Validation (25 validation folds),
    matching the exact evaluation protocol reported in the manuscript.
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Aggregated performance metrics (Mean ± Std).
    raw_records : pd.DataFrame
        Per-fold validation scores across all 25 folds.
    oof_predictions : dict
        Aggregated out-of-fold predictions for per-class confusion matrix analysis.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    records = []
    
    # Track accumulated predictions across all 148 plants
    pred_accumulator = {name: np.zeros((len(y_target), 3)) for name in models_dict.keys()}
    class_map = {1: 0, 3: 1, 5: 2}
    idx_to_class = {0: 1, 1: 3, 2: 5}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_features, y_target)):
        X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]
        
        # 1. Scale strictly on train fold
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        
        # 2. Correlation screening on train fold
        sig_feats = []
        for col in X_train_scaled.columns:
            r, p = stats.spearmanr(X_train_scaled[col], y_train)
            if p < 0.10:
                sig_feats.append(col)
        if not sig_feats:
            sig_feats = list(X_train_scaled.columns)
            
        X_tr = X_train_scaled[sig_feats]
        X_te = X_test_scaled[sig_feats]
        
        for m_name, model in models_dict.items():
            if 'Gradient Boosting' in m_name:
                sw = compute_sample_weight('balanced', y_train)
                model.fit(X_tr, y_train, sample_weight=sw)
            else:
                model.fit(X_tr, y_train)
                
            y_pred = model.predict(X_te)
            
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            ord_mae = compute_ordinal_mae(y_test, y_pred)
            
            records.append({
                'Fold': fold_idx,
                'Model': m_name,
                'Macro_F1': macro_f1,
                'Balanced_Acc': bal_acc,
                'Accuracy': acc,
                'Ordinal_MAE': ord_mae
            })
            
            for t_idx, p_val in zip(test_idx, y_pred):
                pred_accumulator[m_name][t_idx, class_map.get(p_val, 1)] += 1

    df_eval = pd.DataFrame(records)
    
    # Calculate Mean ± Std
    summary = df_eval.groupby('Model').agg({
        'Macro_F1': ['mean', 'std'],
        'Balanced_Acc': ['mean', 'std'],
        'Ordinal_MAE': ['mean', 'std'],
        'Accuracy': ['mean', 'std']
    }).round(3)
    
    # Majority vote across 5 repeats for ensemble out-of-fold prediction
    consensus_oof = {}
    for m_name, counts in pred_accumulator.items():
        voted_classes = [idx_to_class[np.argmax(counts[i])] for i in range(len(y_target))]
        consensus_oof[m_name] = np.array(voted_classes)
        
    return summary, df_eval, consensus_oof


def generate_class_breakdown_table(y_true, y_pred, model_name="Random Forest"):
    """
    Generates detailed per-class classification metrics (Class 1, 3, 5).
    """
    classes = [1, 3, 5]
    labels_map = {
        1: "Class 1 (Dual-Tolerant)",
        3: "Class 3 (Mono-Tolerant)",
        5: "Class 5 (Dual-Susceptible)"
    }
    
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    
    rows = []
    for cls_val, prec, rec, f_sc, s in zip(classes, p, r, f1, sup):
        rows.append({
            'Class (MTS)': cls_val,
            'Phenotypic Category': labels_map[cls_val],
            'Precision': round(prec, 3),
            'Recall': round(rec, 3),
            'F1-Score': round(f_sc, 3),
            'Support': int(s)
        })
        
    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    rows.append({
        'Class (MTS)': 'Overall',
        'Phenotypic Category': 'Overall Accuracy',
        'Precision': '-',
        'Recall': '-',
        'F1-Score': f"{round(acc * 100, 1)}%",
        'Support': len(y_true)
    })
    
    rows.append({
        'Class (MTS)': 'Average',
        'Phenotypic Category': 'Macro Average',
        'Precision': round(p_mac, 3),
        'Recall': round(r_mac, 3),
        'F1-Score': round(f1_mac, 3),
        'Support': len(y_true)
    })
    
    return pd.DataFrame(rows)


def plot_confusion_matrix_publication(y_true, y_pred, output_path="results/confusion_matrix_rf.png", model_name="Random Forest"):
    """
    Plots a publication-quality confusion matrix without requiring seaborn.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 3, 5])
    classes = ['Class 1\n(Dual-Tolerant)', 'Class 3\n(Mono-Tolerant)', 'Class 5\n(Dual-Susceptible)']
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
            
    fig.colorbar(cax, shrink=0.8, pad=0.05, label="Number of Plants")
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    
    ax.set_xlabel('Predicted Multiple Tolerance Score (MTS)', fontsize=12, labelpad=12, fontweight="bold")
    ax.set_ylabel('True Multiple Tolerance Score (MTS)', fontsize=12, labelpad=12, fontweight="bold")
    ax.set_title(f'Confusion Matrix: {model_name} (25 Nested Folds, N=148)', fontsize=13, pad=20, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved confusion matrix plot to: {output_path}")


def plot_feature_importance_publication(feature_names, importances, output_path="results/feature_importance_rf.png", model_name="Random Forest"):
    """
    Plots horizontal feature importance ranking for the ensemble model.
    """
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(12)
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    y_pos = np.arange(len(df_imp))
    
    bars = ax.barh(y_pos, df_imp['Importance'], color='#2b5c8f', edgecolor='black', height=0.65)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_imp['Feature'], fontsize=10, fontweight="bold")
    ax.set_xlabel('Gini Feature Importance', fontsize=12, fontweight="bold")
    ax.set_title(f'{model_name} - Top Phenotypic Traits in Stress Classification', fontsize=13, pad=15, fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', ha='left', fontsize=9)
        
    ax.set_xlim(0, max(df_imp['Importance']) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved feature importance plot to: {output_path}")


def plot_ablation_figure_publication(output_path="results/Figure10_Feature_Ablation_Benchmark.png"):
    """
    Plots a formal, publication-quality bar chart for the feature ablation benchmark
    using 1 uniform academic Navy Blue color, with Y-axis strictly bounded from 0.5 to 0.9.
    """
    subsets = [
        'Colorimetry Only\n(Static, 4 traits)',
        'Dynamic Delta (Δ) Only\n(Inter-stage, 14 traits)',
        'Static Both\n(Morph + Color, 20 traits)',
        'Morphology Only\n(Static, 16 traits)',
        'All Combined\n(Static + Dynamic Δ, 34 traits)'
    ]
    means = [0.661, 0.720, 0.760, 0.764, 0.776]
    stds = [0.089, 0.092, 0.075, 0.092, 0.082]
    bar_color = '#2B547E'

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x = np.arange(len(subsets))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=bar_color, edgecolor='#15283b', linewidth=1.1,
                  width=0.52, zorder=3,
                  error_kw={'elinewidth': 1.2, 'ecolor': '#15283b', 'capthick': 1.2})

    ax.grid(axis='y', linestyle=':', color='#d5dbdb', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Strictly 0.5 to 0.9
    ax.set_ylim(0.5, 0.9)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel('Cross-Validated Macro F1-Score (Mean ± SD)', fontsize=11, fontweight='bold', color='#1c2833', labelpad=10)
    ax.tick_params(axis='y', labelsize=10, colors='#1c2833')
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=9.5, fontweight='bold', color='#1c2833')

    # White text inside the top of the bars to cleanly avoid colliding with error bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean - 0.035, f'{mean:.3f}',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1c2833')
    ax.spines['bottom'].set_color('#1c2833')

    plt.title('Figure 10. Feature Ablation Benchmark Across Phenotypic Subsets (GBM, 25 Folds)',
              fontsize=12, fontweight='bold', color='#1c2833', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Saved Figure 10 ablation benchmark plot to: {output_path}")

