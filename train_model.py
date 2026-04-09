import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def prepare_data(sn):
    file_path = f'preprocessed_{sn}.csv'
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Feature Engineering
    for lag in [1, 3, 6, 12]:
        df[f'Pac_lag{lag}'] = df['Pac'].shift(lag)
        df[f'light_lag{lag}'] = df['light_intensity'].shift(lag)
        
    df['Pac_roll_mean_1h'] = df['Pac'].rolling(window=12).mean()
    df['Pac_roll_std_1h'] = df['Pac'].rolling(window=12).std()
    df['Pac_roll_std_15min'] = df['Pac'].rolling(window=3).std()
    
    df['light_diff'] = df['light_intensity'].diff()
    df['light_accel'] = df['light_diff'].diff()
    df['Pac_diff'] = df['Pac'].diff()
    
    df['target_delta'] = df['Pac'].shift(-12) - df['Pac']
    df['hour'] = df['time'].dt.hour
    
    weather_features = ['tmp_weather', 'reh_weather', 'ws_weather', 'rain_weather', 'sky', 'pty']
    features_all = [
        'hour', 'Pac', 'Pac_diff',
        'Pac_lag1', 'Pac_lag3', 'Pac_lag6', 'Pac_lag12',
        'light_intensity', 'light_lag1', 'light_diff', 'light_accel',
        'Pac_roll_mean_1h', 'Pac_roll_std_1h', 'Pac_roll_std_15min',
        'Upv1', 'Ipv1', 'Upv2', 'Ipv2', 'Upv3', 'Ipv3', 
        'Tmod', 'Tamb', 'PF',
        'temperature', 'humidity', 'wind_speed', 'uv_index'
    ] + weather_features
    
    print(f"Initial DF rows: {len(df)}")
    df_clean = df.dropna(subset=['target_delta'] + features_all).copy()
    print(f"Cleaned DF rows: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("DEBUG: Checking NaNs per column:")
        print(df[['target_delta'] + features_all].isna().sum())
    
    # 2026-03-08 Fixed Date Split
    test_start_date = '2026-03-08'
    train_df = df_clean[df_clean['time'] < test_start_date]
    test_df = df_clean[df_clean['time'] >= test_start_date]
    
    return train_df, test_df, features_all, weather_features

def train_model(X_train, y_train, X_test, y_test, features, name, sample_weight=None, depth=6):
    if len(X_train) == 0:
        return np.array([]), np.array([])
    print(f"  Training {name}...")
    X_tr = X_train[features]
    X_te = X_test[features]
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=depth, random_state=42, n_jobs=-1, early_stopping_rounds=50)
    
    val_split = int(len(X_tr) * 0.9)
    X_t, X_v = X_tr.iloc[:val_split], X_tr.iloc[val_split:]
    y_t, y_v = y_train.iloc[:val_split], y_train.iloc[val_split:]
    sw_t = sample_weight.iloc[:val_split] if sample_weight is not None else None
    sw_v = sample_weight.iloc[val_split:] if sample_weight is not None else None
    
    model.fit(X_t, y_t, sample_weight=sw_t, eval_set=[(X_v, y_v)], sample_weight_eval_set=[sw_v], verbose=False)
    
    pred_delta = model.predict(X_te)
    actual_current_pac = X_test['Pac'].values
    pred_pac = np.maximum(actual_current_pac + pred_delta, 0)
    actual_future_pac = actual_current_pac + y_test.values
    
    return pred_pac, actual_future_pac

def run_comparison(sn):
    print(f"\n===== [Weather Integrated Comparison] Inverter {sn} =====")
    train_df, test_df, features_all, weather_features = prepare_data(sn)
    
    if len(train_df) == 0:
        print(f"Skipping {sn} due to empty data.")
        return

    y_train = train_df['target_delta']
    y_test = test_df['target_delta']
    
    results = {}
    feat_a = ['hour', 'Pac', 'light_intensity', 'temperature', 'humidity', 'wind_speed', 'uv_index', 
              'Upv1', 'Ipv1', 'Upv2', 'Ipv2', 'Upv3', 'Ipv3', 'Tmod', 'Tamb', 'PF'] + weather_features
    pred_a, actual = train_model(train_df, y_train, test_df, y_test, feat_a, "A. Baseline (Weather)")
    results['Baseline'] = pred_a
    
    feat_b = [f for f in features_all if f not in ['light_diff', 'light_accel']]
    pred_b, _ = train_model(train_df, y_train, test_df, y_test, feat_b, "B. Memory-Enhanced (Weather)")
    results['Memory'] = pred_b
    
    max_delta = np.max(np.abs(y_train))
    sw = 1.0 + 4.0 * (np.abs(y_train) / max_delta)
    pred_c, _ = train_model(train_df, y_train, test_df, y_test, features_all, "C. Edge-Tracking (Weather)", sample_weight=sw, depth=9)
    results['Edge'] = pred_c
    
    summary = []
    for name, pred in results.items():
        if len(pred) > 0:
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            summary.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    
    pred_pers = test_df['Pac'].values
    summary.append({'Model': 'Persistence', 'MAE': mean_absolute_error(actual, pred_pers), 
                    'RMSE': np.sqrt(mean_squared_error(actual, pred_pers)), 'R2': r2_score(actual, pred_pers)})
    
    print("\nPerformance Summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    
    plt.figure(figsize=(15, 8))
    start_idx, end_idx = 400, 650
    plt.plot(actual[start_idx:end_idx], label='Actual', color='black')
    plt.plot(results['Baseline'][start_idx:end_idx], label='A. Baseline', alpha=0.5)
    plt.plot(results['Memory'][start_idx:end_idx], label='B. Memory', alpha=0.6)
    plt.plot(results['Edge'][start_idx:end_idx], label='C. Edge', color='red', linestyle='--')
    plt.legend(); plt.grid(True)
    plt.savefig(f'final_comparison_{sn}_weather.png'); plt.close()

if __name__ == "__main__":
    for sn in ['3012', '5009']:
        run_comparison(sn)