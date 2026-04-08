import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

kma_df = pd.read_csv('KMA_weather.csv', encoding='cp949')
kma_df['일시'] = pd.to_datetime(kma_df['일시'])
kma_df['시간'] = kma_df['일시'].dt.hour

kma_daytime = kma_df[(kma_df['시간'] >= 6) & (kma_df['시간'] <= 18) & (kma_df['일사(MJ/m2)'] > 0)].copy()

target_col = '일사(MJ/m2)'

# Case 1: 강수량 제외, 전운량 포함
feature_cols_1 = ['기온(°C)', '습도(%)', '해면기압(hPa)', '풍속(m/s)', '전운량(10분위)']

# Case 2: 강수량, 전운량 모두 제외 (원래 사용)
feature_cols_2 = ['기온(°C)', '습도(%)', '해면기압(hPa)', '풍속(m/s)']

print("=== 데이터 비교 ===\n")

for case, features in [(1, feature_cols_1), (2, feature_cols_2)]:
    kma_clean = kma_daytime[features + [target_col]].dropna()
    
    X = kma_clean[features]
    y = kma_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Case {case}: {', '.join(features)}")
    print(f"  학습 데이터: {len(kma_clean):,}개")
    print(f"  Test R²: {test_score:.4f}\n")
    
    # 특성 중요도
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("  특성 중요도:")
    for idx, row in importance_df.iterrows():
        print(f"    {row['feature']:15} : {row['importance']:.4f}")
    print()
