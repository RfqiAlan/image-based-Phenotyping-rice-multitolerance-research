import numpy as np
import pandas as pd
from scipy import stats
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class IBPFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Leakage-safe statistical feature selector combining:
    1. Spearman Rank Correlation (p < 0.05)
    2. Exploratory Factor Analysis (EFA) with Kaiser-Guttman criterion (eigenvalue > 1)
    3. Communality Filtering (h^2 >= communality_threshold, default 0.50)
    """
    def __init__(self, communality_threshold=0.50):
        self.communality_threshold = communality_threshold
        self.selected_columns_ = None
        self.communalities_ = None
        self.factor_loadings_ = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()
            
        cols = X_df.columns.tolist()
        
        # 1. Spearman Rank Correlation Filter with target if y is provided
        passed_spearman = []
        if y is not None:
            y_arr = np.array(y)
            for col in cols:
                corr, p_val = stats.spearmanr(X_df[col].values, y_arr)
                if p_val < 0.05:
                    passed_spearman.append(col)
        else:
            passed_spearman = cols
            
        if len(passed_spearman) < 2:
            passed_spearman = cols

        # 2. Factor Analysis & Communality Filtering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_df[passed_spearman].dropna())
        
        # Eigenvalue determination
        fa_initial = FactorAnalyzer(rotation=None, n_factors=len(passed_spearman))
        fa_initial.fit(scaled_data)
        eigenvalues, _ = fa_initial.get_eigenvalues()
        n_factors = max(1, sum(eigenvalues > 1.0))
        
        # Varimax rotated Factor Analysis
        fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
        fa.fit(scaled_data)
        
        communalities = np.sum(fa.loadings_**2, axis=1)
        self.communalities_ = dict(zip(passed_spearman, communalities))
        self.factor_loadings_ = fa.loadings_
        
        # Filter traits with communality >= threshold
        selected = [feat for feat, comm in self.communalities_.items() if comm >= self.communality_threshold]
        if not selected:
            selected = passed_spearman
            
        self.selected_columns_ = selected
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_columns_].values
        return X[:, [i for i, col in enumerate(X.columns) if col in self.selected_columns_]]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
