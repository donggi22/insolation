# 일사량 및 발전량 예측 프로젝트

## 프로젝트 개요
이 프로젝트는 기상 데이터와 IoT 센서 데이터를 활용하여 태양광 발전량을 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다. MOA 기상 데이터를 학습용으로 사용하고, IoT 센서 데이터를 실시간 예측에 적용합니다.

## 주요 목표
- 기상변수로부터 일사량 예측
- IoT 센서 데이터를 활용한 발전량 예측
- 다양한 모델 비교 및 성능 평가
- 실무 적용 가능한 예측 모델 개발

## 데이터 구조

### 원본 데이터
1. **moa_weather.csv**: MOA 기상 데이터 (학습용)
   - 일사량(MJ/m2): 실제 측정값 (타겟)
   - 기상변수들: 기온, 습도, 기압, 풍속, 전운량 등

2. **total_IoT_sensor.csv**: IoT 센서 데이터 (예측용)
   - 기상변수들: 온도, 습도, 기압, 풍속 등
   - 일사량 센서 없음 → 모델로 예측

3. **total_power.csv**: 인버터 발전량 데이터
   - Pac: 출력 발전량
   - Upv1-3, Ipv1-3: 스트링별 전압/전류
   - Tmod, Tamb: 모듈 및 주변 온도
   - PF: 역률

### 전처리된 데이터
- **preprocessed_3012.csv**: 인버터 3012 데이터 (24,230행)
- **preprocessed_5009.csv**: 인버터 5009 데이터 (24,178행)
- 센서와 인버터 데이터를 5분 단위로 통합

## 파일 구조
- **EDA 노트북**:
  - `power_eda.ipynb`: 발전량 데이터 탐색
  - `weather_eda.ipynb`: 기상 데이터 탐색
  - `sonsor_eda.ipynb`: 센서 데이터 탐색

- **스크립트 파일**:
  - `preprocess_data.py`: 데이터 전처리 스크립트
  - `train_model.py`: 모델 훈련 스크립트

- **결과 파일**:
  - `final_metrics.txt`: 최종 모델 메트릭스
  - `비교 결과.txt`: 모델 비교 결과
  - `iot_insolation_predictions_clean.csv`: 예측 결과

- **preview/**: 추가 실험 및 모델
  - `hourly_forecast_xgboost.py`: XGBoost 기반 시간별 예측
  - `hourly_insolation_forecast_pytorch.py`: PyTorch 기반 일사량 예측
  - `improved_insolation_model.py`: 개선된 모델
  - 기타 테스트 파일들

## 접근 방법
1. **데이터 탐색**: 각 데이터의 구조와 특성 파악 (EDA 노트북)
2. **데이터 전처리**: 센서와 인버터 데이터 통합 및 정제
3. **특성 선택**: IoT 센서가 제공하는 변수들 위주로 선택
4. **모델 개발**: Baseline, Memory-Enhanced, Edge-Tracking 모델 등 다양한 접근
5. **성능 평가**: MAE, RMSE, R² 지표로 평가
6. **예측 적용**: IoT 데이터에 모델 적용

## 설치 및 실행

### 가상환경 설정
```bash
conda create -n insolation python=3.10
conda activate insolation
pip install -r requirements.txt
```

### 데이터 전처리
```bash
python preprocess_data.py
```

### 모델 훈련
```bash
python train_model.py
```

### EDA 실행
Jupyter Notebook에서 각 EDA 파일을 실행:
- `jupyter notebook power_eda.ipynb`
- `jupyter notebook weather_eda.ipynb`
- `jupyter notebook sonsor_eda.ipynb`

## 모델 성능 결과

### Weather Integrated 모델 비교 (인버터 3012)
| 모델 | MAE | RMSE | R² |
|------|-----|------|----|
| Baseline | 3.39 | 5.25 | 0.88 |
| Memory-Enhanced | 3.14 | 4.99 | 0.89 |
| Edge-Tracking | 3.33 | 5.28 | 0.87 |
| Persistence | 6.60 | 8.39 | 0.68 |

### Weather Integrated 모델 비교 (인버터 5009)
| 모델 | MAE | RMSE | R² |
|------|-----|------|----|
| Baseline | 3.06 | 4.70 | 0.88 |
| Memory-Enhanced | 2.85 | 4.48 | 0.89 |
| Edge-Tracking | 2.92 | 4.68 | 0.88 |
| Persistence | 5.93 | 7.57 | 0.68 |

## 다음 단계
- 모델 최적화 및 추가 특성 엔지니어링
- 실시간 예측 시스템 구현
- 추가 데이터 수집 및 모델 재훈련