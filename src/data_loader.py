import os
import pandas as pd
import numpy as np

def load_and_preprocess_data(data_dir="data"):
    """
    Loads drought and salinity phenotypic datasets, verifies row-level plant identity,
    computes static, dynamic delta, and multi-stress tolerance targets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing Kekeringan_Gabung.csv and Salinitas_Gabung.csv.
        
    Returns:
    --------
    dict containing:
        - 'X': Combined static features DataFrame (20 features: 10 Drought + 10 Salin)
        - 'X_all': Combined static + dynamic delta features DataFrame
        - 'y_mts': Ordinal/Multiclass target array (Classes 1, 3, 5)
        - 'y_binary': Binary target array (1: Dual-Tolerant, 0: Non-Dual-Tolerant)
        - 'feature_names': List of feature names
        - 'df_drought': Raw drought DataFrame
        - 'df_salinity': Raw salinity DataFrame
    """
    drought_path = os.path.join(data_dir, "Kekeringan_Gabung.csv")
    salinity_path = os.path.join(data_dir, "Salinitas_Gabung.csv")
    
    if not os.path.exists(drought_path) or not os.path.exists(salinity_path):
        raise FileNotFoundError(f"Datasets not found in {data_dir}. Ensure Kekeringan_Gabung.csv and Salinitas_Gabung.csv exist.")
        
    df_k = pd.read_csv(drought_path)
    df_s = pd.read_csv(salinity_path)
    
    # Verify plant identity row by row
    assert (df_k.iloc[:, 0] == df_s.iloc[:, 0]).all(), "Plant Genotype IDs do not match row-for-row!"
    n_plants = len(df_k)
    print(f"[DataLoader] Successfully verified and aligned {n_plants} plants across sequential stress phases.")
    
    # Standardize target columns
    if 'Luma_Y' in df_k.columns: df_k = df_k.rename(columns={'Luma_Y': 'SES_Score'})
    if 'Luma_Y' in df_s.columns: df_s = df_s.rename(columns={'Luma_Y': 'SES_Score'})
    
    ses_k = df_k['SES_Score'].values
    ses_s = df_s['SES_Score'].values
    
    # Multi-Stress Tolerance Score (MTS) Mapping
    # 1 = Dual-Tolerant (SES <= 3 in both)
    # 3 = Mono-Tolerant (SES <= 3 in one phase only)
    # 5 = Dual-Susceptible (SES > 3 in both phases)
    def map_mts(k, s):
        if k <= 3 and s <= 3:
            return 1
        elif (k <= 3 and s > 3) or (k > 3 and s <= 3):
            return 3
        else:
            return 5

    y_mts = np.array([map_mts(k, s) for k, s in zip(ses_k, ses_s)])
    y_binary = np.array([1 if (k <= 3 and s <= 3) else 0 for k, s in zip(ses_k, ses_s)])
    
    # Base extracted features (Collinearity-adjusted: dropped VARI due to r=0.99 with NGRDI)
    base_features = ['Area', 'Perim', 'Widht', 'Height', 'Major', 'Minor', 'Circle', 'Solidity', 'GLI', 'NGRDI']
    
    X_k = df_k[base_features].add_suffix('_Drought')
    X_s = df_s[base_features].add_suffix('_Salin')
    X_static = pd.concat([X_k, X_s], axis=1)
    
    # Inter-stage Dynamic Change Features (Delta and Ratios)
    df_delta = pd.DataFrame(index=df_k.index)
    eps = 1e-6
    for trait in ['Area', 'Perim', 'Height', 'Widht', 'Solidity', 'GLI', 'NGRDI']:
        df_delta[f'Delta_{trait}'] = df_s[trait] - df_k[trait]
        df_delta[f'Ratio_{trait}'] = (df_s[trait] + eps) / (df_k[trait] + eps)
        
    X_all = pd.concat([X_static, df_delta], axis=1)
    
    return {
        'X': X_static,
        'X_all': X_all,
        'y_mts': y_mts,
        'y_binary': y_binary,
        'df_drought': df_k,
        'df_salinity': df_s,
        'feature_names': list(X_static.columns)
    }
