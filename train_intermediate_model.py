import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_intermediate_models():
    print("\n===== Stage 1: Training Intermediate Feature Models (Light & Tmod) =====")
    df = pd.read_csv('preprocessed_daily_ver3.csv')
    
    # We only need one SN's weather/sensor data for the common environment
    # Weather is the same for both, sensors are typically shared or similar
    # Let's just group by date to get unique daily weather/sensor rows
    df_env = df.groupby('date').agg({
        'tmp_mean': 'mean', 'tmp_max': 'mean', 'tmp_min': 'mean',
        'reh_mean': 'mean', 'rain_sum': 'mean', 'sky_mean': 'mean', 'sky_afternoon': 'mean',
        'light_mean': 'mean', 'light_max': 'mean', 'light_afternoon': 'mean',
        'Tmod_mean': 'mean' 
    }).reset_index()

    # Drop NaNs before training
    df_env = df_env.dropna().copy()

    features = ['tmp_mean', 'tmp_max', 'reh_mean', 'rain_sum', 'sky_mean', 'sky_afternoon']
    
    # Add seasonal feature
    df_env['month'] = pd.to_datetime(df_env['date']).dt.month

    X = df_env[features + ['month']]
    
    targets = ['light_mean', 'Tmod_mean']
    
    predictions = pd.DataFrame({'date': df_env['date']})
    
    for target in targets:
        print(f"  Training model for: {target}")
        y = df_env[target]
        
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbosity=-1, random_state=42)
        
        # Cross-validation to get "Predicted" values for the training set without overfilling
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(df_env))
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_preds[val_idx] = model.predict(X_val)
            
        mae = mean_absolute_error(y, oof_preds)
        r2 = r2_score(y, oof_preds)
        print(f"    {target} CV Results -> MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Fit final model on all data for future use
        model.fit(X, y)
        joblib.dump(model, f'model_{target}.pkl')
        
        predictions[f'pred_{target}'] = oof_preds

    predictions.to_csv('intermediate_predictions.csv', index=False)
    print("Stage 1 Complete! Saved intermediate_predictions.csv and models.")

if __name__ == "__main__":
    train_intermediate_models()
