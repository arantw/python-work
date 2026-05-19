import form
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager

fontManager.addfont("ChineseFont.ttf")
mpl.rc('font', family='ChineseFont')

# 1. 讀取原始資料
df = pd.read_csv('data.csv')
df.columns = [c.strip() for c in df.columns]

# 2. 定義函數：剔除斜線與日期（如 '23.2/29' -> 23.2），僅抽取溫度數值
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

# 應用清洗
df['Max_Temp'] = df['最高/日期'].apply(extract_temp)
df['Min_Temp'] = df['最低/日期'].apply(extract_temp)

# 3. 計算每年全台測站的平均高溫與平均低溫
annual_avg_extremes = df.groupby('年份').agg(
    Annual_Avg_Max=('Max_Temp', 'mean'),
    Annual_Avg_Min=('Min_Temp', 'mean')
).reset_index()


# 4. 繪製帶有趨勢線（Trendlines）的暖化分析圖
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(annual_avg_extremes['年份'], annual_avg_extremes['Annual_Avg_Max'], color='orangered', marker='o', label='年均高點 (早上)')
ax.plot(annual_avg_extremes['年份'], annual_avg_extremes['Annual_Avg_Min'], color='dodgerblue', marker='s', label='年平均低點 (晚上)')

# 計算並繪製白天高溫的線性迴歸趨勢線
z_amax = np.polyfit(annual_avg_extremes['年份'], annual_avg_extremes['Annual_Avg_Max'], 1)
p_amax = np.poly1d(z_amax)
ax.plot(annual_avg_extremes['年份'], p_amax(annual_avg_extremes['年份']), "r--", alpha=0.7, label=f'平均高趨勢 (slope=+{z_amax[0]:.3f}°C/year)')


# 計算並繪製夜晚低溫的線性迴歸趨勢線
z_amin = np.polyfit(annual_avg_extremes['年份'], annual_avg_extremes['Annual_Avg_Min'], 1)
p_amin = np.poly1d(z_amin)
ax.plot(annual_avg_extremes['年份'], p_amin(annual_avg_extremes['年份']), "b--", alpha=0.7, label=f'平均低趨勢 (slope=+{z_amin[0]:.3f}°C/year)')


# 圖表美化設定
ax.set_title('年平均最高和最低氣溫（2009-2025 年）', fontsize=14)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('溫度 (°C)', fontsize=12)
ax.set_xticks(annual_avg_extremes['年份'])
ax.set_xticklabels(annual_avg_extremes['年份'], rotation=45)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='center right')
plt.tight_layout()

# 儲存圖表
plt.savefig('cruve.png')
plt.close()