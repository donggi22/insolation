import pandas as pd
import numpy as np
from datetime import datetime

def rain_num(df_weather):
    df_weather['rn1_num'] = df_weather['rn1'].replace('강수없음', '0')
    df_weather['rn1_num'] = df_weather['rn1_num'].replace('1mm 미만', '0.1')

    mask = df_weather['rn1_num'] == '\\N'
    df_weather.loc[mask & (df_weather['pty'] == 0), 'rn1_num'] = '0'
    df_weather.loc[mask & (df_weather['pty'] != 0), 'rn1_num'] = '0.1'

    df_weather['rn1_num'] = df_weather['rn1_num'].str.replace('mm', '')
    df_weather['rn1_num'] = pd.to_numeric(df_weather['rn1_num'], errors='coerce')

    return df_weather


def preprocess():
    print("Starting data preprocessing...")

    # =========================
    # 1. Load Inverter Data
    # =========================
    print("Loading total_power.csv...")
    df_power = pd.read_csv('total_power.csv', engine='python')

    df_power['SN'] = df_power['SN'].astype(str)
    df_power = df_power[df_power['SN'].str.endswith(('3012', '5009'))].copy()

    df_power['SN_short'] = df_power['SN'].apply(
        lambda x: '3012' if x.endswith('3012') else '5009'
    )

    df_power['datetime'] = pd.to_datetime(df_power['crt_dttm'])
    df_power['time_5min'] = df_power['datetime'].dt.floor('5min')

    # 🔥 이상치 처리 (groupby 전에!)
    print("Handling inverter outliers...")

    # Ipv → 0
    for col in ['Ipv1', 'Ipv2', 'Ipv3']:
        df_power.loc[df_power[col] > 100, col] = 0

    # Tmod, Tamb → 100 초과인 이상치 제거
    df_power = df_power[
        (df_power['Tmod'] <= 100) &
        (df_power['Tamb'] <= 100)
    ]

    # PF 이상치 여부 feature
    df_power['PF_abnormal'] = (df_power['PF'] > 1).astype(int)

    # PF → NaN
    df_power.loc[df_power['PF'] > 1, 'PF'] = np.nan

    power_cols = [
        'time_5min', 'SN_short',
        'Pac', 'Upv1', 'Ipv1', 'Upv2', 'Ipv2', 'Upv3', 'Ipv3',
        'Tmod', 'Tamb', 'PF', 'PF_abnormal'
    ]

    df_power = (
        df_power[power_cols]
        .groupby(['time_5min', 'SN_short'])
        .mean()
        .reset_index()
    )

    # =========================
    # 2. Sensor Data
    # =========================
    print("Loading sensor data...")
    df_sensor = pd.read_csv('total_IoT_sensor.csv', engine='python')

    df_sensor['datetime'] = pd.to_datetime(df_sensor['create_at'])
    df_sensor['time_5min'] = df_sensor['datetime'].dt.floor('5min')

    sensor_features = [
        'light_intensity', 'temperature',
        'humidity', 'wind_speed', 'uv_index'
    ]

    df_sensor_5min = (
        df_sensor.groupby('time_5min')[sensor_features]
        .mean()
        .reset_index()
    )

    # 👉 센서는 그대로 (가볍게만)
    df_sensor_5min[sensor_features] = (
        df_sensor_5min[sensor_features]
        .interpolate()
        .fillna(0)
    )

    # =========================
    # 3. Weather Data
    # =========================
    print("Loading weather data...")
    df_weather = pd.read_csv('moa_weather.csv', engine='python')

    df_weather = rain_num(df_weather)

    df_weather['tm_str'] = (
        df_weather['base_date'].astype(str)
        + df_weather['base_time'].astype(str).str.zfill(4)
    )

    df_weather['tm'] = pd.to_datetime(df_weather['tm_str'], format='%Y%m%d%H%M')

    # 🔥 정렬 + 마지막 값 유지
    df_weather = df_weather.sort_values('tm').groupby('tm').last()

    # 🔥 이상치 제거
    df_weather = df_weather[
        (df_weather['t1h'] >= 0) &
        (df_weather['reh'] >= 0) &
        (df_weather['wsd'] >= 0) &
        (df_weather['sky'] >= 0) &
        (df_weather['pty'] >= 0)
    ]

    # 5분 리샘플
    new_index = pd.date_range(
        start=df_weather.index.min(),
        end=df_weather.index.max(),
        freq='5min'
    )

    df_weather_5min = df_weather.reindex(new_index)

    # 연속형
    cont_cols = ['t1h', 'reh', 'wsd']

    df_weather_5min[cont_cols] = df_weather_5min[cont_cols].interpolate(
        method='linear',
        limit=12
    )

    df_weather_5min['rn1_num'] = df_weather_5min['rn1_num'].fillna(0)

    # 범주형
    df_weather_5min[['sky', 'pty']] = (
        df_weather_5min[['sky', 'pty']]
        .ffill()
        .bfill()
    )

    df_weather_5min = (
        df_weather_5min
        .reset_index()
        .rename(columns={'index': 'time_5min'})
    )

    df_weather_5min = df_weather_5min.rename(columns={
        't1h': 'tmp_weather',
        'reh': 'reh_weather',
        'wsd': 'ws_weather',
        'rn1_num': 'rain_weather'
    })

    weather_final_cols = [
        'time_5min',
        'tmp_weather', 'reh_weather',
        'ws_weather', 'rain_weather',
        'sky', 'pty'
    ]

    df_weather_5min = df_weather_5min[weather_final_cols]

    # =========================
    # 4. Merge
    # =========================
    print("Merging...")
    df_merged = pd.merge(df_power, df_sensor_5min, on='time_5min', how='left')
    df_merged = pd.merge(df_merged, df_weather_5min, on='time_5min', how='left')

    # =========================
    # 5. Save
    # =========================
    for sn in ['3012', '5009']:
        df_sn = df_merged[df_merged['SN_short'] == sn].copy()

        df_sn = df_sn.drop(columns=['SN_short']).rename(columns={'time_5min': 'time'})

        # 👉 마지막 결측만 정리
        df_sn = df_sn.ffill().bfill().fillna(0)

        df_sn.to_csv(f'preprocessed_{sn}_ver2.csv', index=False)
        print(f"Saved preprocessed_{sn}_ver2.csv")

    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess()