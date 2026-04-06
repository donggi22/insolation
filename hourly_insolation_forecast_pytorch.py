import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== PyTorch 기반 1시간 단위 시계열 예측 모델 (1시간 뒤 일사량 예측) ===\n")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

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

# 사용할 피처와 타겟 컬럼 정의
features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'uv_index', 'light_intensity', '기온(°C)', '습도(%)', '해면기압(hPa)', '풍속(m/s)']
target = '일사(MJ/m2)'
numeric_cols = features + [target]

# 1시간 단위로 리샘플링 (평균값 사용, 숫자형 컬럼만)
merged_df.set_index('create_at', inplace=True)
hourly_df = merged_df[numeric_cols].resample('h').mean().dropna()

print(f"1시간 집계 후 데이터 크기: {hourly_df.shape}")
print(f"시간 범위: {hourly_df.index.min()} ~ {hourly_df.index.max()}")

# 낮시간만 필터링 (일사량 의미있는 시간)
hourly_df['hour'] = hourly_df.index.hour
hourly_daytime = hourly_df[(hourly_df['hour'] >= 6) & (hourly_df['hour'] <= 18)]

print(f"낮시간 데이터: {hourly_daytime.shape}")

# 2. 시퀀스 데이터 생성
print("\n2. 시퀀스 데이터 생성...")

# 시퀀스 길이: 6시간 과거 데이터로 다음 1시간 예측
seq_length = 6

def create_sequences(data, seq_length, target_col):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target_col].iloc[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(hourly_daytime, seq_length, target)
print(f"시퀀스 데이터 크기: X={X.shape}, y={y.shape}")

# 데이터 분할 (시간 순서 유지)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# 스케일링
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# X 스케일링 (3D 배열이므로 reshape)
X_train_reshaped = X_train.reshape(-1, len(features))
X_test_reshaped = X_test.reshape(-1, len(features))

X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 3. PyTorch 데이터셋 및 데이터로더
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 마지막 시퀀스 출력
        out = self.fc(out)
        return out

model = LSTMModel(input_size=len(features), hidden_size=64, num_layers=2, output_size=1).to(device)
print(f"\n모델 구조:\n{model}")

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 모델 학습
print("\n5. 모델 학습...")
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 6. 성능 평가
print("\n6. 성능 평가...")
model.eval()
y_pred_scaled = []
y_test_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        y_pred_scaled.extend(outputs.cpu().numpy().flatten())
        y_test_list.extend(y_batch.cpu().numpy().flatten())

y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(np.array(y_test_list).reshape(-1, 1)).flatten()

r2 = r2_score(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(((y_test_original - y_pred) ** 2).mean())

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f} MJ/m²")
print(f"RMSE: {rmse:.4f} MJ/m²")

# 7. 시각화
print("\n7. 예측 결과 시각화...")
plt.figure(figsize=(15, 5))

# 학습 곡선
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='학습 손실')
plt.plot(val_losses, label='검증 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.title('학습 곡선')
plt.legend()
plt.grid(True, alpha=0.3)

# 예측 vs 실제
plt.subplot(1, 3, 2)
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', linewidth=2)
plt.xlabel('실제 일사량 (MJ/m²)')
plt.ylabel('예측 일사량 (MJ/m²)')
plt.title(f'예측 vs 실제 (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)

# 시간별 예측
plt.subplot(1, 3, 3)
plt.plot(y_test_original[:100], label='실제', alpha=0.7)
plt.plot(y_pred[:100], label='예측', alpha=0.7)
plt.xlabel('시간 스텝')
plt.ylabel('일사량 (MJ/m²)')
plt.title('시간별 예측 비교 (첫 100개)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hourly_forecast_pytorch_results.png', dpi=100, bbox_inches='tight')
print("결과 플롯 저장: hourly_forecast_pytorch_results.png")

# 8. 미래 예측 예시
print("\n8. 미래 예측 예시...")
# 최근 6시간 데이터로 다음 1시간 예측
recent_data = hourly_daytime[features].iloc[-seq_length:].values.reshape(1, seq_length, len(features))
recent_scaled = scaler_X.transform(recent_data.reshape(-1, len(features))).reshape(recent_data.shape)

model.eval()
with torch.no_grad():
    recent_tensor = torch.tensor(recent_scaled, dtype=torch.float32).to(device)
    future_pred_scaled = model(recent_tensor).cpu().numpy().flatten()[0]

future_pred = scaler_y.inverse_transform(np.array([[future_pred_scaled]])).flatten()[0]

print(f"최근 6시간 데이터 기반 다음 1시간 일사량 예측: {future_pred:.3f} MJ/m²")
print(f"예측 시간대: {hourly_daytime.index[-1] + pd.Timedelta(hours=1)}")

print("\n=== PyTorch 기반 1시간 뒤 예측 모델 학습 완료 ===")
print("이 모델은 과거 6시간 IoT/KMA 데이터를 사용해 다음 1시간 일사량을 예측합니다.")