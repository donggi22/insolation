import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def prepare_daily_data(sn):
    df = pd.read_csv('preprocessed_daily.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['SN_short'] == int(sn)].copy().sort_values('date')
    
    # 1. Targets: Tomorrow and Day-after-tomorrow energy
    df['target_t1'] = df['Pac_sum'].shift(-1)
    df['target_t2'] = df['Pac_sum'].shift(-2)
    
    # 2. History Features
    df['Pac_lag1'] = df['Pac_sum'].shift(1)
    df['Pac_ma3'] = df['Pac_sum'].rolling(window=3).mean()
    
    # 3. Future Weather Features (The core idea)
    # We use tomorrow's and day-after's weather as features for "today"
    weather_cols = [
        'tmp_mean', 'tmp_max', 'tmp_daytime', 
        'sky_mean', 'sky_daytime', 'sky_afternoon', 
        'reh_mean', 'rain_sum', 'light_afternoon'
    ]
    
    for col in weather_cols:
        df[f'{col}_t1'] = df[col].shift(-1)
        df[f'{col}_t2'] = df[col].shift(-2)
    
    # 4. Temporal
    df['month'] = df['date'].dt.month
    
    # Feature list
    features = [
        'month', 'Pac_sum', 'Pac_lag1', 'Pac_ma3'
    ] + [f'{col}_t1' for col in weather_cols] + [f'{col}_t2' for col in weather_cols]
    
    # Drop rows with NaNs (due to shifts)
    df_clean = df.dropna(subset=['target_t1', 'target_t2'] + features).copy()
    
    # Split: Keep last 15 days for testing
    test_days = 15
    train_df = df_clean.iloc[:-test_days]
    test_df = df_clean.iloc[-test_days:]
    
    return train_df, test_df, features

def train_daily_model(sn):
    print(f"\n===== Training Daily Model for Inverter {sn} =====")
    train_df, test_df, features = prepare_daily_data(sn)
    
    if len(train_df) < 5:
        print("Not enough data to train.")
        return

    X_train = train_df[features]
    y_train = train_df[['target_t1', 'target_t2']]
    
    X_test = test_df[features]
    y_test = test_df[['target_t1', 'target_t2']]
    
    # LightGBM Multi-output
    # Using MultiOutputRegressor because natively LGBM target is usually 1D
    base_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        importance_type='gain',
        verbosity=-1
    )
    
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test) # Result is (N, 2)
    
    # Metrics
    targets = ['t+1 Day', 't+2 Day']
    summary = []
    for i, target_name in enumerate(targets):
        y_true = y_test.iloc[:, i]
        y_pred = preds[:, i]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        summary.append({
            'Target': target_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
    
    print("\nPerformance Summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(15, 6))
    
    # t+1
    plt.subplot(1, 2, 1)
    plt.plot(test_df['date'], y_test.iloc[:, 0], 'o-', label='Actual', color='black')
    plt.plot(test_df['date'], preds[:, 0], 's--', label='Predicted', color='blue')
    plt.title(f'Inverter {sn} - Tomorrow Forecast (t+1)')
    plt.xticks(rotation=45)
    plt.legend(); plt.grid(True)
    
    # t+2
    plt.subplot(1, 2, 2)
    plt.plot(test_df['date'], y_test.iloc[:, 1], 'o-', label='Actual', color='black')
    plt.plot(test_df['date'], preds[:, 1], 's--', label='Predicted', color='red')
    plt.title(f'Inverter {sn} - Day-After Forecast (t+2)')
    plt.xticks(rotation=45)
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'daily_forecast_result_{sn}.png')
    plt.close()
    print(f"Saved visualization to daily_forecast_result_{sn}.png")

    # 7. Save Actual vs Predicted to CSV
    results_df = pd.DataFrame({
        'date': test_df['date'],
        'actual_t1': y_test.iloc[:, 0].values,
        'pred_t1': preds[:, 0],
        'actual_t2': y_test.iloc[:, 1].values,
        'pred_t2': preds[:, 1]
    })
    results_csv = f'forecast_results_{sn}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Saved numerical results to {results_csv}")

    # Feature Importance (for one of the targets)
    # Just to see what's driving the model
    importances = model.estimators_[0].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
    print("\nTop 10 Features (t+1):")
    print(feat_imp)

if __name__ == "__main__":
    for sn in ['3012', '5009']:
        train_daily_model(sn)
