import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def prepare_pipeline_data(sn):
    # 1. Load Preprocessed Data
    df_main = pd.read_csv('preprocessed_daily_ver3.csv')
    df_main['date'] = pd.to_datetime(df_main['date'])
    df_main = df_main[df_main['SN_short'] == int(sn)].copy().sort_values('date')
    
    # 2. Load Intermediate Predictions (Stage 1 Output)
    df_pred = pd.read_csv('intermediate_predictions.csv')
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    
    # 3. Merge
    df = pd.merge(df_main, df_pred, on='date', how='left')
    
    # 4. Create Targets (Tomorrow & Day-after DYield)
    df['target_t1'] = df['DYield_today'].shift(-1)
    df['target_t2'] = df['DYield_today'].shift(-2)
    
    # 5. Lag Features
    df['DYield_lag1'] = df['DYield_today'].shift(1)
    df['DYield_ma3'] = df['DYield_today'].rolling(window=3).mean()
    
    # 6. Future Stage 1 Features (Shift pred_light, pred_Tmod for t+1 and t+2)
    intermediate_cols = ['pred_light_mean', 'pred_Tmod_mean']
    for col in intermediate_cols:
        df[f'{col}_t1'] = df[col].shift(-1)
        df[f'{col}_t2'] = df[col].shift(-2)
        
    # 7. Future Weather Features (KMA Forecast)
    weather_cols = ['tmp_mean', 'sky_mean', 'rain_sum']
    for col in weather_cols:
        df[f'{col}_t1'] = df[col].shift(-1)
        df[f'{col}_t2'] = df[col].shift(-2)

    df['month'] = df['date'].dt.month
    
    # Feature List
    features = [
        'month', 'DYield_today', 'DYield_lag1', 'DYield_ma3'
    ] + [f'{col}_t1' for col in intermediate_cols + weather_cols] \
      + [f'{col}_t2' for col in intermediate_cols + weather_cols]
      
    df_clean = df.dropna(subset=['target_t1', 'target_t2'] + features).copy()
    
    # Split
    test_days = 20
    train_df = df_clean.iloc[:-test_days]
    test_df = df_clean.iloc[-test_days:]
    
    return train_df, test_df, features

def train_power_pipeline(sn):
    print(f"\n===== Stage 2: Training Power Pipeline Model for Inverter {sn} =====")
    train_df, test_df, features = prepare_pipeline_data(sn)
    
    if len(train_df) < 5:
        print("Empty training set.")
        return

    X_train = train_df[features]
    y_train = train_df[['target_t1', 'target_t2']]
    
    X_test = test_df[features]
    y_test = test_df[['target_t1', 'target_t2']]
    
    model = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42, verbosity=-1
    ))
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Metrics
    summary = []
    for i, target_name in enumerate(['t+1 Day', 't+2 Day']):
        mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
        r2 = r2_score(y_test.iloc[:, i], preds[:, i])
        summary.append({'Target': target_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    
    print(pd.DataFrame(summary).to_string(index=False))
    
    # Save Results
    res_df = pd.DataFrame({
        'date': test_df['date'],
        'actual_t1': y_test.iloc[:, 0].values,
        'pred_t1': preds[:, 0],
        'actual_t2': y_test.iloc[:, 1].values,
        'pred_t2': preds[:, 1]
    })
    res_df.to_csv(f'pipeline_results_{sn}.csv', index=False)
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1,2,1)
    plt.plot(test_df['date'], y_test.iloc[:, 0], 'o-', label='Actual', color='black')
    plt.plot(test_df['date'], preds[:, 0], 's--', label='Predicted', color='blue')
    plt.title(f'SN {sn} - Tomorrow Forecast (t+1)')
    plt.legend(); plt.grid(True); plt.xticks(rotation=45)
    
    plt.subplot(1,2,2)
    plt.plot(test_df['date'], y_test.iloc[:, 1], 'o-', label='Actual', color='black')
    plt.plot(test_df['date'], preds[:, 1], 's--', label='Predicted', color='red')
    plt.title(f'SN {sn} - Day-After Forecast (t+2)')
    plt.legend(); plt.grid(True); plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'pipeline_plot_{sn}.png')
    plt.close()

if __name__ == "__main__":
    for sn in ['3012', '5009']:
        train_power_pipeline(sn)
