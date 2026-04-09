import pandas as pd
import numpy as np
from datetime import datetime

def preprocess():
    print("Starting data recovery and preprocessing (Python Engine Fix)...")

    # 1. Load Inverter Data
    print("Loading total_power.csv...")
    df_power = pd.read_csv('total_power.csv', engine='python')
    df_power['SN'] = df_power['SN'].astype(str)
    df_power = df_power[df_power['SN'].str.endswith(('3012', '5009'))].copy()
    df_power['SN_short'] = df_power['SN'].apply(lambda x: '3012' if x.endswith('3012') else '5009')
    df_power['datetime'] = pd.to_datetime(df_power['crt_dttm'])
    df_power['time_5min'] = df_power['datetime'].dt.floor('5min')
    power_cols = ['time_5min', 'SN_short', 'Pac', 'Upv1', 'Ipv1', 'Upv2', 'Ipv2', 'Upv3', 'Ipv3', 'Tmod', 'Tamb', 'PF']
    df_power = df_power[power_cols].groupby(['time_5min', 'SN_short']).mean().reset_index()

    # 2. Load Sensor Data
    print("Loading sensor data...")
    df_sensor = pd.read_csv('total_IoT_sensor.csv', engine='python')
    df_sensor['datetime'] = pd.to_datetime(df_sensor['create_at'])
    df_sensor['time_5min'] = df_sensor['datetime'].dt.floor('5min')
    sensor_features = ['light_intensity', 'temperature', 'humidity', 'wind_speed', 'uv_index']
    df_sensor_5min = df_sensor.groupby('time_5min')[sensor_features].mean().reset_index()
    df_sensor_5min[sensor_features] = df_sensor_5min[sensor_features].interpolate(method='linear').fillna(0)

    # 3. Load Weather Data
    print("Loading weather data...")
    df_weather = pd.read_csv('moa_weather.csv', engine='python')
    if 'rn1' in df_weather.columns:
        df_weather['rn1'] = df_weather['rn1'].replace('강수없음', '0.0')
    
    df_weather['tm_str'] = df_weather['base_date'].astype(str) + df_weather['base_time'].astype(str).str.zfill(4)
    df_weather['tm'] = pd.to_datetime(df_weather['tm_str'], format='%Y%m%d%H%M')
    df_weather = df_weather.sort_values('tm').drop_duplicates('tm').set_index('tm')
    
    new_index = pd.date_range(start=df_weather.index.min(), end=df_weather.index.max(), freq='5min')
    df_weather_5min = df_weather.reindex(new_index)
    
    for col in ['t1h', 'reh', 'wsd', 'rn1', 'sky', 'pty']:
        if col in df_weather_5min.columns:
            df_weather_5min[col] = pd.to_numeric(df_weather_5min[col], errors='coerce')

    cont_cols = ['t1h', 'reh', 'wsd', 'rn1']
    df_weather_5min[cont_cols] = df_weather_5min[cont_cols].interpolate(method='linear').fillna(0)
    df_weather_5min[['sky', 'pty']] = df_weather_5min[['sky', 'pty']].ffill().fillna(0)
    
    df_weather_5min = df_weather_5min.reset_index().rename(columns={'index': 'time_5min'})
    df_weather_5min = df_weather_5min.rename(columns={'t1h': 'tmp_weather', 'reh': 'reh_weather', 'wsd': 'ws_weather', 'rn1': 'rain_weather'})
    weather_final_cols = ['time_5min', 'tmp_weather', 'reh_weather', 'ws_weather', 'rain_weather', 'sky', 'pty']
    df_weather_5min = df_weather_5min[weather_final_cols]

    # 4. Merge
    print("Merging...")
    df_merged = pd.merge(df_power, df_sensor_5min, on='time_5min', how='left')
    df_merged = pd.merge(df_merged, df_weather_5min, on='time_5min', how='left')
    
    # 5. Save
    for sn in ['3012', '5009']:
        df_sn = df_merged[df_merged['SN_short'] == sn].copy()
        df_sn = df_sn.drop(columns=['SN_short']).rename(columns={'time_5min': 'time'})
        df_sn = df_sn.fillna(method='ffill').fillna(method='bfill').fillna(0)
        df_sn.to_csv(f'preprocessed_{sn}.csv', index=False)
        print(f"Saved preprocessed_{sn}.csv")

    print("Data recovery complete!")

if __name__ == "__main__":
    preprocess()