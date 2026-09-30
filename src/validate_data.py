from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'raw'/'city_day.csv'
if not DATA.exists(): raise FileNotFoundError(f'Missing {DATA}')
df=pd.read_csv(DATA)
required=['City','Date','AQI']
missing=[c for c in required if c not in df.columns]
if missing: raise ValueError(f'Missing columns: {missing}')
print('[SUCCESS] Air Quality dataset validation passed.')
print(f'[INFO] Shape: {df.shape}')
print(f'[INFO] Duplicate rows: {df.duplicated().sum()}')
print(f'[INFO] AQI missing values: {df["AQI"].isna().sum()}')
