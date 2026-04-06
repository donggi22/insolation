import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== XGBoost 기반 1시간 단위 시계열 예측 모델 (1시간 뒤 일사량 예측) ===\n")

# 1. 데이터 로드 및 전처리
print("1. 데이터 로드 및 전처리...")
kma_df = pd.read_csv('KMA_weather.csv', encoding='cp949')
iot_df = pd.read_csv('total_iot_sensor.csv')

# 시간 변환
kma_df['일시'] = pd.to_datetime(kma_df['일시'])
iot_df['create_at'] = pd.to_datetime(iot_df['create_at'])

# 데이터 병합
kma_subset = kma_df[['일시', '일사(MJ/m2)', '기온(°C)', '습도(%)', '해면기압(hPa)', '풍속(m/s)']].dropna()
merged_df = pd.merge_asof(
    iot_df.sort_values('create_at'),
    kma_subset.sort_values('일시'),
    left_on='create_at',
    right_on='일시',
    direction='nearest'
)

# 1시간 단위로 리샘플링 (평균값 사용)
merged_df.set_index('create_at', inplace=True)
# 숫자형 컬럼만 선택
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
hourly_df = merged_df[numeric_cols].resample('h').mean().dropna()

print(f"1시간 집계 후 데이터 크기: {hourly_df.shape}")
print(f"시간 범위: {hourly_df.index.min()} ~ {hourly_df.index.max()}")

# 낮시간만 필터링 (일사량 의미있는 시간)
hourly_df['hour'] = hourly_df.index.hour
hourly_daytime = hourly_df[(hourly_df['hour'] >= 6) & (hourly_df['hour'] <= 18)]

print(f"낮시간 데이터: {hourly_daytime.shape}")

# 2. 시계열 피처 생성 (과거 6시간 데이터로 다음 1시간 예측)
print("\n2. 시계열 피처 생성...")
# 사용할 기본 피처
base_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'uv_index', 'light_intensity', '기온(°C)', '습도(%)', '해면기압(hPa)', '풍속(m/s)']

# 과거 6시간 데이터를 피처로 확장
lookback_hours = 6
feature_cols = []

for hour in range(1, lookback_hours + 1):
    for feature in base_features:
        feature_cols.append(f'{feature}_t-{hour}')

# 타겟: 다음 1시간 일사량
target_col = '일사(MJ/m2)_t+1'

# 데이터프레임 재구성
data = []
for i in range(lookback_hours, len(hourly_daytime)):
    row = {}
    # 과거 데이터
    for hour in range(1, lookback_hours + 1):
        past_idx = i - hour
        for feature in base_features:
            row[f'{feature}_t-{hour}'] = hourly_daytime[feature].iloc[past_idx]

    # 미래 타겟
    row[target_col] = hourly_daytime['일사(MJ/m2)'].iloc[i]

    data.append(row)

forecast_df = pd.DataFrame(data)
print(f"시계열 피처 데이터 크기: {forecast_df.shape}")
print(f"피처 수: {len(feature_cols)}")

# 3. 데이터 분할 및 스케일링
print("\n3. 데이터 분할 및 스케일링...")
X = forecast_df[feature_cols]
y = forecast_df[target_col]

# 시간 순서 유지하며 분할
split_idx = int(len(forecast_df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. XGBoost 모델 학습
print("\n4. XGBoost 모델 학습...")
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 5. 성능 평가
print("\n5. 성능 평가...")
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(((y_test - y_pred) ** 2).mean())

print(f"학습 R²: {train_score:.4f}")
print(f"테스트 R²: {test_score:.4f}")
print(f"MAE: {mae:.4f} MJ/m²")
print(f"RMSE: {rmse:.4f} MJ/m²")

# 시간 기반 교차 검증
print("\n시간 기반 교차 검증...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for train_idx, val_idx in tscv.split(X_train_scaled):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    fold_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    fold_model.fit(X_fold_train, y_fold_train)
    fold_score = fold_model.score(X_fold_val, y_fold_val)
    cv_scores.append(fold_score)

print(f"CV 평균 R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# 6. 피처 중요도
print("\n6. 피처 중요도...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("상위 피처:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25}: {row['importance']:.4f}")

# 7. 시각화
print("\n7. 예측 결과 시각화...")
plt.figure(figsize=(15, 5))

# 예측 vs 실제
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('실제 일사량 (MJ/m²)')
plt.ylabel('예측 일사량 (MJ/m²)')
plt.title(f'예측 vs 실제 (R² = {test_score:.3f})')
plt.grid(True, alpha=0.3)

# 오차 분포
plt.subplot(1, 3, 2)
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('예측 오차 (MJ/m²)')
plt.ylabel('빈도')
plt.title(f'오차 분포 (평균: {errors.mean():.3f})')
plt.grid(True, alpha=0.3)

# 시간별 예측
plt.subplot(1, 3, 3)
plt.plot(y_test.values[:100], label='실제', alpha=0.7)
plt.plot(y_pred[:100], label='예측', alpha=0.7)
plt.xlabel('시간 스텝')
plt.ylabel('일사량 (MJ/m²)')
plt.title('시간별 예측 비교 (첫 100개)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hourly_forecast_xgboost_results.png', dpi=100, bbox_inches='tight')
print("결과 플롯 저장: hourly_forecast_xgboost_results.png")

# 8. 미래 예측 예시
print("\n8. 미래 예측 예시...")
# 최근 6시간 데이터로 다음 1시간 예측
recent_features = []
for hour in range(1, lookback_hours + 1):
    past_idx = len(hourly_daytime) - hour
    for feature in base_features:
        recent_features.append(hourly_daytime[feature].iloc[past_idx])

recent_df = pd.DataFrame([recent_features], columns=feature_cols)
recent_scaled = scaler.transform(recent_df)

future_pred = model.predict(recent_scaled)[0]

print(f"최근 6시간 데이터 기반 다음 1시간 일사량 예측: {future_pred:.3f} MJ/m²")
print(f"예측 시간대: {hourly_daytime.index[-1] + pd.Timedelta(hours=1)}")

print("\n=== XGBoost 기반 1시간 뒤 예측 모델 학습 완료 ===")
print("이 모델은 과거 6시간 IoT/KMA 데이터를 사용해 다음 1시간 일사량을 예측합니다.")