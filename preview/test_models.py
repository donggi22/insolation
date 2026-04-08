import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

kma_df = pd.read_csv('KMA_weather.csv', encoding='cp949')
kma_df['일시'] = pd.to_datetime(kma_df['일시'])
kma_df['시간'] = kma_df['일시'].dt.hour

kma_daytime = kma_df[(kma_df['시간'] >= 6) & (kma_df['시간'] <= 18) & (kma_df['일사(MJ/m2)'] > 0)].copy()

target_col = '일사(MJ/m2)'
feature_cols = ['기온(°C)', '습도(%)', '해면기압(hPa)', '강수량(mm)', '풍속(m/s)']

kma_clean = kma_daytime[feature_cols + [target_col]].dropna()

X = kma_clean[feature_cols]
y = kma_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 여러 모델 비교
models = {
    'Ridge (강정규화)': Ridge(alpha=10.0),
    'Lasso (L1정규화)': Lasso(alpha=0.01),
    'RandomForest (얕음)': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    'GradientBoosting (약함)': GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma=0.01)
}

print("=== 모델 비교 ===\n")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    print(f"{name:20} | Test R²: {test_score:7.4f} | CV R²: {cv_scores:7.4f}")
