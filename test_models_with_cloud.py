import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

kma_df = pd.read_csv('KMA_weather.csv', encoding='cp949')
kma_df['일시'] = pd.to_datetime(kma_df['일시'])
kma_df['시간'] = kma_df['일시'].dt.hour

kma_daytime = kma_df[(kma_df['시간'] >= 6) & (kma_df['시간'] <= 18) & (kma_df['일사(MJ/m2)'] > 0)].copy()

target_col = '일사(MJ/m2)'

# 전운량 포함한 특성들
feature_cols_with_cloud = ['기온(°C)', '습도(%)', '해면기압(hPa)', '강수량(mm)', 
                           '풍속(m/s)', '전운량(10분위)']

kma_clean = kma_daytime[feature_cols_with_cloud + [target_col]].dropna()

print(f"전운량이 포함된 학습 데이터: {len(kma_clean):,}개\n")

X = kma_clean[feature_cols_with_cloud]
y = kma_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 여러 모델 비교 (전운량 포함)
models = {
    'Ridge': Ridge(alpha=10.0),
    'RandomForest (depth=3)': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    'RandomForest (depth=5)': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
}

print("=== 전운량 포함한 모델 비교 ===\n")
best_model = None
best_score = -1

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    print(f"{name:25} | Test R²: {test_score:7.4f} | CV R²: {cv_scores:7.4f}")
    
    if test_score > best_score:
        best_score = test_score
        best_model = (name, model)

print(f"\n✓ 최고 성능: {best_model[0]} (R² = {best_score:.4f})")

# 특성 중요도
if hasattr(best_model[1], 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols_with_cloud,
        'importance': best_model[1].feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n특성 중요도:")
    print(importance_df)
