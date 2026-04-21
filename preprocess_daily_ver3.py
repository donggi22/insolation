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

def preprocess_daily_ver3():
    print("Starting daily data preprocessing (Ver 3 - DYield Based)...")

    # =========================
    # 1. Load Inverter Data (DYield)
    # =========================
    print("Loading total_power.csv...")
    df_power = pd.read_csv('total_power.csv', engine='python')
    df_power['SN'] = df_power['SN'].astype(str)
    df_power = df_power[df_power['SN'].str.endswith(('3012', '5009'))].copy()
    df_power['SN_short'] = df_power['SN'].apply(lambda x: '3012' if x.endswith('3012') else '5009')
    # Using crt_dttm as it seems to be the record time
    df_power['datetime'] = pd.to_datetime(df_power['crt_dttm'])
    df_power['date'] = df_power['datetime'].dt.date

    # Outlier handling (Tmod, Tamb)
    df_power = df_power[(df_power['Tmod'] <= 80) & (df_power['Tamb'] <= 60)]

    # Daily aggregation (DYield: MAX is the total for the day)
    # Tmod: Mean for the day
    df_power_daily = df_power.groupby(['date', 'SN_short']).agg({
        'DYield': 'max',
        'Tmod': 'mean',
        'PF': 'mean'
    }).reset_index()
    
    # Rename for clarity
    df_power_daily = df_power_daily.rename(columns={'DYield': 'DYield_today', 'Tmod': 'Tmod_mean', 'PF': 'PF_mean'})

    # =========================
    # 2. Sensor Data (Compressed)
    # =========================
    print("Loading sensor data...")
    df_sensor = pd.read_csv('total_IoT_sensor.csv', engine='python')
    df_sensor['datetime'] = pd.to_datetime(df_sensor['create_at'])
    df_sensor['date'] = df_sensor['datetime'].dt.date
    df_sensor['hour'] = df_sensor['datetime'].dt.hour
    
    sensor_features = ['light_intensity', 'temperature']
    df_sensor_daily = df_sensor.groupby('date')[sensor_features].agg(['mean', 'max']).reset_index()
    df_sensor_daily.columns = ['date', 'light_mean', 'light_max', 'temp_sensor_mean', 'temp_sensor_max']

    # Afternoon Focus
    df_sensor_afternoon = df_sensor[(df_sensor['hour'] >= 12) & (df_sensor['hour'] < 18)]
    df_sensor_afternoon = df_sensor_afternoon.groupby('date')[['light_intensity']].mean().reset_index()
    df_sensor_afternoon = df_sensor_afternoon.rename(columns={'light_intensity': 'light_afternoon'})

    df_sensor_final = pd.merge(df_sensor_daily, df_sensor_afternoon, on='date', how='left').fillna(0)

    # =========================
    # 3. Weather Data (KMA)
    # =========================
    print("Loading weather data...")
    df_weather = pd.read_csv('moa_weather.csv', engine='python')
    df_weather = rain_num(df_weather)
    df_weather['tm_str'] = df_weather['base_date'].astype(str) + df_weather['base_time'].astype(str).str.zfill(4)
    df_weather['tm'] = pd.to_datetime(df_weather['tm_str'], format='%Y%m%d%H%M')
    df_weather['date'] = df_weather['tm'].dt.date
    df_weather['hour'] = df_weather['tm'].dt.hour

    # Detailed Weather Summary
    df_weather_daily = df_weather.groupby('date').agg({
        't1h': ['mean', 'max', 'min'],
        'reh': 'mean',
        'rn1_num': 'sum',
        'sky': 'mean'
    }).reset_index()
    df_weather_daily.columns = ['date', 'tmp_mean', 'tmp_max', 'tmp_min', 'reh_mean', 'rain_sum', 'sky_mean']

    # Afternoon sky
    df_weather_afternoon = df_weather[(df_weather['hour'] >= 12) & (df_weather['hour'] < 18)]
    df_weather_afternoon = df_weather_afternoon.groupby('date')[['sky']].mean().reset_index()
    df_weather_afternoon = df_weather_afternoon.rename(columns={'sky': 'sky_afternoon'})

    df_weather_final = pd.merge(df_weather_daily, df_weather_afternoon, on='date', how='left').fillna(0)

    # =========================
    # 4. Final Merge
    # =========================
    print("Merging everything...")
    df_merged = pd.merge(df_power_daily, df_sensor_final, on='date', how='left')
    df_merged = pd.merge(df_merged, df_weather_final, on='date', how='left')

    df_merged['date'] = df_merged['date'].astype(str)
    df_merged.to_csv('preprocessed_daily_ver3.csv', index=False)
    print("Preprocessing complete! Saved preprocessed_daily_ver3.csv")

if __name__ == "__main__":
    preprocess_daily_ver3()
