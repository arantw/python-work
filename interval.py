import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager

fontManager.addfont("ChineseFont.ttf")
mpl.rc('font', family='ChineseFont')

df = pd.read_csv('data.csv')
df.columns = [c.strip() for c in df.columns]


def extract_temp(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if '/' in val_str:
        return float(val_str.split('/')[0])
    try:
        return float(val_str)
    except ValueError:
        return np.nan

df['Max_Temp'] = df['最高/日期'].apply(extract_temp)
df['Min_Temp'] = df['最低/日期'].apply(extract_temp)

monthly_extremes = df.groupby(['年份', '月份']).agg(
    Max_Temperature=('Max_Temp', 'max'),
    Min_Temperature=('Min_Temp', 'min')
).reset_index()

monthly_extremes.columns = ['Year', 'Month', 'Max_Temperature', 'Min_Temperature']
monthly_extremes.to_csv('weather_monthly_extremes.csv', index=False)

monthly_extremes['Date'] = pd.to_datetime(
    monthly_extremes['Year'].astype(str) + '-' + monthly_extremes['Month'].astype(str) + '-01'
)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(monthly_extremes['Date'], monthly_extremes['Max_Temperature'],
        color='crimson', label='每月最高溫度', linewidth=1.5, marker='.')

ax.plot(monthly_extremes['Date'], monthly_extremes['Min_Temperature'],
        color='royalblue', label='每月最低溫', linewidth=1.5, marker='.')

ax.fill_between(monthly_extremes['Date'],
                monthly_extremes['Min_Temperature'],
                monthly_extremes['Max_Temperature'],
                color='purple', alpha=0.1, label='溫度範圍')

ax.set_title('月度溫度範圍（2009-2025）', fontsize=14)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('溫度 (°C)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('interval.png')
plt.close()
