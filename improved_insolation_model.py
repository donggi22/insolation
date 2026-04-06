import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== IoT 센서 기반 일사량 예측 모델 개선 버전 ===\n")

# 1. 데이터 로드
print("1. 데이터 로드 중...")
kma_df = pd.read_csv('KMA_weather.csv', encoding='cp949')
iot_df = pd.read_csv('total_iot_sensor.csv')

print(f"KMA 데이터 크기: {kma_df.shape}")
print(f"IoT 데이터 크기: {iot_df.shape}")

# 2. 데이터 전처리 및 결합
print("\n2. 데이터 전처리 및 결합...")
# 시간 컬럼 변환
kma_df['일시'] = pd.to_datetime(kma_df['일시'])
iot_df['create_at'] = pd.to_datetime(iot_df['create_at'])

# 시간별로 병합 (가장 가까운 시간 매칭)
kma_subset = kma_df[['일시', '일사(MJ/m2)', '기온(°C)', '습도(%)', '해면기압(hPa)', '풍속(m/s)']].dropna()
merged_df = pd.merge_asof(
    iot_df.sort_values('create_at'),
    kma_subset.sort_values('일시'),
    left_on='create_at',
    right_on='일시',
    direction='nearest'
)

print(f"병합 후 데이터 크기: {merged_df.shape}")

# 3. 피처 엔지니어링
print("\n3. 피처 엔지니어링...")
# 시간 기반 피처
merged_df['hour'] = merged_df['create_at'].dt.hour
merged_df['month'] = merged_df['create_at'].dt.month
merged_df['season'] = merged_df['month'] % 12 // 3  # 0: 겨울, 1: 봄, 2: 여름, 3: 가을

# 낮시간 데이터만 필터링 (일사량 의미있는 시간)
merged_df = merged_df[(merged_df['hour'] >= 6) & (merged_df['hour'] <= 18) & (merged_df['일사(MJ/m2)'] > 0)]

# 사용할 피처 선택 (IoT 센서 기반)
features = [
    'temperature', 'humidity', 'pressure', 'wind_speed',
    'uv_index', 'light_intensity', 'hour', 'season'
]
target = '일사(MJ/m2)'

# 결측치 제거
clean_df = merged_df[features + [target]].dropna()
print(f"정제 후 데이터 크기: {clean_df.shape}")

# 4. 데이터 분할
print("\n4. 데이터 분할 및 스케일링...")
X = clean_df[features]
y = clean_df[target]

# 시간 순서 유지하며 분할 (시계열 데이터 고려)
split_idx = int(len(clean_df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"학습 데이터: {X_train_scaled.shape}, 테스트 데이터: {X_test_scaled.shape}")

# 5. 모델 학습
print("\n5. 모델 학습 (XGBoost)...")
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 6. 성능 평가
print("\n6. 성능 평가...")
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
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv)
print(f"CV 평균 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. 피처 중요도
print("\n7. 피처 중요도...")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("상위 피처:")
for idx, row in feature_importance.head(8).iterrows():
    print(f"  {row['feature']:15}: {row['importance']:.4f}")

# 8. 시각화
print("\n8. 예측 결과 시각화...")
plt.figure(figsize=(12, 4))

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

# 피처 중요도
plt.subplot(1, 3, 3)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('중요도')
plt.title('피처 중요도')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improved_model_results.png', dpi=100, bbox_inches='tight')
print("결과 플롯 저장: improved_model_results.png")

# 9. IoT 데이터만으로 예측 예시
print("\n9. IoT 데이터 예측 예시...")
# 테스트 데이터의 첫 번째 샘플로 예측
sample_iot = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample_iot)
prediction = model.predict(sample_scaled)[0]
actual = y_test.iloc[0]

print(f"샘플 IoT 데이터: {sample_iot.to_dict('records')[0]}")
print(f"예측 일사량: {prediction:.3f} MJ/m²")
print(f"실제 일사량: {actual:.3f} MJ/m²")
print(f"오차: {abs(prediction - actual):.3f} MJ/m²")

print("\n=== 모델 학습 완료 ===")
print("이 모델을 사용하여 새로운 IoT 센서 데이터로 일사량을 예측할 수 있습니다.")