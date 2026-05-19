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
    if pd.isna(val) or '/' not in str(val): return np.nan
    return float(str(val).split('/')[0])

df['Max_Temp'] = df['最高/日期'].apply(extract_temp)
df['Min_Temp'] = df['最低/日期'].apply(extract_temp)

annual_data = df.groupby('年份').agg(
    Avg_Mean=('平均', 'mean'),
    Avg_Max=('Max_Temp', 'mean'),
    Avg_Min=('Min_Temp', 'mean')
).reset_index()

years_hist = annual_data['年份'].values
future_years = np.array([2026, 2027, 2028])
all_years = np.concatenate([years_hist, future_years])


p_max = np.poly1d(np.polyfit(years_hist, annual_data['Avg_Max'], 1))
p_min = np.poly1d(np.polyfit(years_hist, annual_data['Avg_Min'], 1))


pred_max = p_max(future_years)
pred_min = p_min(future_years)

fig, ax = plt.subplots(figsize=(12, 6))


ax.plot(years_hist, annual_data['Avg_Max'], color='orangered', marker='o', label='歷史平均最高值')
ax.plot(years_hist, annual_data['Avg_Min'], color='dodgerblue', marker='s', label='歷史平均最低值')

ax.plot(future_years, pred_max, color='red', marker='x', linestyle='--', linewidth=2, label='預測平均最高 (2026-2028)')
ax.plot(future_years, pred_min, color='blue', marker='x', linestyle='--', linewidth=2, label='預測平均最低 (2026-2028)')

ax.plot([years_hist[-1], future_years[0]], [annual_data['Avg_Max'].iloc[-1], pred_max[0]], color='orangered', linestyle=':')
ax.plot([years_hist[-1], future_years[0]], [annual_data['Avg_Min'].iloc[-1], pred_min[0]], color='dodgerblue', linestyle=':')

ax.set_title('未來三年（2026-2028 年）氣溫預測', fontsize=14)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('溫度 (°C)', fontsize=12)
ax.set_xticks(all_years)
ax.set_xticklabels(all_years, rotation=45)
ax.grid(True, linestyle='--', alpha=0.5)
ax.axvspan(2025.5, 2028.5, color='gray', alpha=0.1, label='預測期')
ax.legend(loc='center right')
plt.tight_layout()

plt.savefig('predict.png')
plt.close()