# 데이터 전처리 완료 보고서

인버터(`total_power.csv`)와 센서(`total_IoT_sensor.csv`) 데이터를 성공적으로 통합하고 전처리를 완료했습니다. 이제 모델링을 위한 준비가 모두 끝났습니다.

## 주요 작업 내역

### 1. 인버터 데이터 정제 (`3012`, `5009`)
- **대상 인버터**: SN 끝자리 `3012`, `5009` 데이터만 추출.
- **피처 보존**: `Upv1-3`, `Ipv1-3` 스트링 데이터를 개별 컬럼으로 유지하였으며, `PF`는 요청대로 원본 스케일(1.0 = 100%)을 유지했습니다.
- **시간 정렬**: 5분 단위(floor)로 정교하게 정렬했습니다.

### 2. 센서 데이터 통합
- **리샘플링**: 1분 단위 센서 데이터를 5분 단위 평균(`mean`)으로 변환하여 인버터 데이터와 주기를 맞췄습니다.
- **주요 피처**: 일사량, 온도, 습도, 풍속, 자외선 지수를 포함했습니다.

### 3. 최종 데이터셋 생성
- 인버터 데이터와 센서 데이터를 타임스탬프 기준으로 병합했습니다.
- **결과 파일**:
    - [preprocessed_3012.csv](file:///c:/Users/greenfesco/Desktop/insolation/preprocessed_3012.csv) (24,230행)
    - [preprocessed_5009.csv](file:///c:/Users/greenfesco/Desktop/insolation/preprocessed_5009.csv) (24,178행)

## 전처리 스크립트
작업에 사용된 파이썬 스크립트입니다: [preprocess_data.py](file:///c:/Users/greenfesco/Desktop/insolation/preprocess_data.py)

---

## 데이터 검증 결과 (`3012` 기준)

| 컬럼명 | 데이터 타입 | 설명 |
| :--- | :--- | :--- |
| `time` | object | 5분 단위 타임스탬프 |
| `Pac` | float64 | 출력 발전량 (Target) |
| `Upv1~3`, `Ipv1~3` | float64 | 인버터 스트링별 전압/전류 |
| `Tmod`, `Tamb` | float64 | 모듈 온도 및 주변 온도 |
| `PF` | float64 | 역률 (원본 유지) |
| `light_intensity` | float64 | 센서 측정 일사량 |
| `humidity` | float64 | 센서 측정 습도 |

> [!NOTE]
> 센서 데이터와 인버터 데이터의 기록 시작 시점이 달라 일부 초기 데이터에 결측치(NaN)가 있을 수 있습니다. 모델링 시 `dropna()` 또는 보간법으로 처리 가능합니다.

이제 이 파일들을 사용하여 **1시간 후 발전량 예측 모델**을 개발할 수 있습니다. 다음 단계로 모델링 코드를 작성할까요?
