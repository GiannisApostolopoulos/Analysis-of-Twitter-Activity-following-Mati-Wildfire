
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
PLOTS_DIR = ROOT / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

# Group tweets by date
grouped_by_date = df.groupby(df['created_at'].dt.date)

# Get the raw daily tweet volume
daily_volume = grouped_by_date.size()

# In some days, tweet volume may be zero. These days are not included in the groupby object, and must be recoreved.
full_range = pd.date_range(
    start=daily_volume.index.min(),
    end=daily_volume.index.max(),
    freq='D'
)
daily_volume = daily_volume.reindex(full_range, fill_value=0)

# Get some useful info
peak_day = daily_volume.idxmax()
peak_volume = daily_volume.max()
inactive_days_count = daily_volume[daily_volume == 0].count()
busiest_days = daily_volume.nlargest(5)
avg_volume = daily_volume.mean()
median_volume = daily_volume.median()

print(f'Day with the highest tweet volume: {peak_day}, with {peak_volume} tweets.')
print(f'Across the dataset, {inactive_days_count} days had zero tweets.')
print(f"The average daily tweet volume is {avg_volume:.2f}, and the median across the dataset is {median_volume:.0f}.")
#
# Visualize daily volume across the whole dataset
plt.figure()
plt.plot(daily_volume.index, daily_volume.values)

plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.title('Tweets per Day')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'daily_volume.png')
plt.show()

# ====================================

# Apply a centered rolling mean of 7 days
rolling_avg = daily_volume.rolling(window=7,min_periods=1, center=True).mean()

# Visualize the rolling mean across the whole dataset
plt.figure()
plt.plot(rolling_avg.index, rolling_avg.values)

plt.xlabel('Date')
plt.ylabel('Avg Number of Tweets')
plt.title('Number of Tweets per Day - Averaged over 7-Day Periods')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rolling_average.png')
plt.show()

# ====================================

# Find the standard deviation for the same rolling windows
rolling_std = daily_volume.rolling(window=7, min_periods=1, center=True).std()

# Convert to dataframe to add another column
daily_volume = daily_volume.to_frame()
daily_volume.index.name = 'Date'
daily_volume.rename(columns={0: 'Tweet Volume'}, inplace=True)

# Identify burst days and add a column that flags burst days with "1", others 0
daily_volume['Is Burst'] = ((daily_volume['Tweet Volume'] > rolling_avg + 1.5 * rolling_std) &
                            (daily_volume['Tweet Volume'] > median_volume)
                            ).astype(int)

burst_count = daily_volume['Is Burst'].sum()

# Logic to add another column that counts the number of consecutive burst days, in order to distinct burst periods
daily_volume['Consecutive Bursts'] = daily_volume['Is Burst'].groupby(
    daily_volume['Is Burst'].ne(daily_volume['Is Burst'].shift()).cumsum()).cumsum().shift(fill_value=0) * (daily_volume['Is Burst'].diff()==-1)


# Create a burst period ID to identify burst periods
daily_volume['Burst Period ID'] = (daily_volume['Is Burst'].diff() == 1).cumsum() * daily_volume['Is Burst']

# Calculate the duration of each burst period
burst_periods = daily_volume[daily_volume['Is Burst'] == 1].groupby('Burst Period ID').size()

# Count the number of burst periods
num_burst_periods = len(burst_periods)

# Calculate the average duration of burst periods
avg_burst_duration = burst_periods.mean()

# Results

print('First 5 burst days:')
for date in daily_volume[daily_volume['Is Burst'] == 1].index[:5]:
    print(date.strftime('%Y-%m-%d'))
print(f'Total burst days: {burst_count}')
print(f'Total burst periods: {num_burst_periods}')
print(f'Average burst duration: {avg_burst_duration}')


# Create the burst visualization plot
plt.figure(figsize=(15, 6))

# Calculate threshold line
threshold = np.maximum(rolling_avg + 1.5 * rolling_std, median_volume)

# Plot the raw volume with conditional coloring
above_threshold = daily_volume['Is Burst'] == 1  # Now using our refined definition

# Plot segments
for i in range(len(daily_volume) - 1):
    color = 'red' if above_threshold.iloc[i] or above_threshold.iloc[i + 1] else 'lightblue'
    plt.plot(daily_volume.index[i:i+2],
             daily_volume['Tweet Volume'].iloc[i:i+2],
             color=color, linewidth=1.5)

# Add threshold line
plt.plot(threshold.index, threshold.values, 'k--', alpha=0.5, linewidth=1, label=f'Threshold (1σ)')
plt.axhline(y=median_volume, color='orange', linestyle=':', alpha=0.7, label=f'Median ({median_volume} tweets)')

plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.title('Daily Tweet Volume with Burst Periods (Requires > Median)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR/'burst_periods.png')
plt.show()


# ====================================

# In later analysis, we will need to load the burst periods. Save them in a separate df

# Get all burst days with their period IDs
bursts = daily_volume[daily_volume['Is Burst'] == 1][['Burst Period ID']].copy()

# In our analysis, we will need the first day of each period.
bursts.drop_duplicates(subset=['Burst Period ID'], inplace=True, keep='first')

# Save the series
bursts.to_csv(DATASETS_DIR / 'bursts.csv', index=True)
