
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
PLOTS_DIR = ROOT / "plots"


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

PLOTS_DIR.mkdir(exist_ok=True)

# Sort the data based on author first, date second
df.sort_values(by=['author_id', 'created_at'], inplace=True)

# Group by author id
grouped_by_author = df.groupby('author_id')

# Get the tweet volume per author
author_volume = grouped_by_author.size()

# Get the time span of activity for each author
activity_duration = grouped_by_author['created_at'].max() - grouped_by_author['created_at'].min()
# Single-tweet authors have zero activity duration. For fair representation in ranking, replace zero with the minimum
# positive duration observed in the dataset:
min_duration = activity_duration[activity_duration > pd.Timedelta(0)].min()
activity_duration[activity_duration == pd.Timedelta(0)] = min_duration

# Create a new df with each user's metrics
author_data = pd.concat([author_volume, activity_duration], axis=1)
author_data.rename(columns={0: 'Tweet Volume', 'created_at': 'Activity Duration'}, inplace=True)

# Get the average interval between tweets for each user and add it to author_data
average_interval = grouped_by_author['created_at'].diff().groupby(df['author_id']).mean()
author_data['Average Interval'] = average_interval


# Some authors show NaT (Not a Time) intervals, while others show zero average intervals
# Hypothesis:
# - NaT intervals -> authors who posted only once (no intervals to calculate)
# - Zero intervals -> authors who posted all their tweets at identical timestamps (perhaps indicating bot activity)
# Verification:

# Get authors with NaT intervals
nat_interval_authors = author_data[author_data['Average Interval'].isna()]
# Verify they all have posted exactly 1 tweet
all_single_tweet = (nat_interval_authors['Tweet Volume'] == 1).all()
# Inverse - get authors with 1 tweet:
single_tweet_authors = author_data[author_data['Tweet Volume'] == 1]
# Verify they all have NaT average interval
all_nat_interval = (single_tweet_authors['Average Interval'].isna()).all()

if not (all_single_tweet and all_nat_interval):
    raise ValueError('Intervals have been calculated wrong.')


# Handle authors with NaT intervals, for fair representation in the rankings
min_interval = author_data[author_data['Average Interval'].notna()].min()
author_data[author_data['Average Interval'].isna()] = min_interval


# ==== Ranking System =====

# Ceil very small intervals to frequency unit
small_interval_filt = author_data['Average Interval'] < pd.Timedelta(hours=1)
author_data.loc[small_interval_filt, 'Average Interval'] = pd.Timedelta(hours=1)

# Extract the hours as int's from the interval column
author_data['Frequency in Hours'] = author_data['Average Interval'] / pd.Timedelta(hours=1)

# Get the score for each user
author_data['Score'] = (0.50 * author_data['Tweet Volume'] +
                       0.25 * author_data['Activity Duration'].dt.days +
                       0.25 * (1/author_data['Frequency in Hours']))

# Create binned bar plots for metrics
tweet_bins = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
tweet_labels = ['1-5', '5-10', '10-25', '25-50', '50-100', '100-250',
                '250-500', '500-1000', '1000-2500', '2500-5000',
                '5000-10000', '10000-20000']
author_data['Tweet Volume Bin'] = pd.cut(author_data['Tweet Volume'],
                                          bins=tweet_bins,
                                          labels=tweet_labels,
                                          right=False)
tweet_counts = author_data['Tweet Volume Bin'].value_counts().sort_index()

duration_bins = [1, 5, 10, 30, 60, 90, 180, 365, 730, 1000, 1600]
duration_labels = ['1-5', '5-10', '10-30', '30-60', '60-90',
                   '90-180', '180-365', '365-730', '730-1000', '1000-1600']
author_data['Duration Days'] = author_data['Activity Duration'].dt.days
author_data['Duration Bin'] = pd.cut(author_data['Duration Days'],
                                      bins=duration_bins,
                                      labels=duration_labels,
                                      right=False)
duration_counts = author_data['Duration Bin'].value_counts().sort_index()

frequency_bins = [1, 2, 4, 6, 12, 24, 48, 72, 168, 336, 720]
frequency_labels = ['1-2', '2-4', '4-6', '6-12', '12-24',
                    '24-48', '48-72', '72-168', '168-336', '336-720']
author_data['Frequency Bin'] = pd.cut(author_data['Frequency in Hours'],
                                       bins=frequency_bins,
                                       labels=frequency_labels,
                                       right=False)
frequency_counts = author_data['Frequency Bin'].value_counts().sort_index()

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.bar(tweet_counts.index, tweet_counts.values, edgecolor='black')
plt.title('Tweet Volume')
plt.xlabel('Number of Tweets')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.bar(duration_counts.index, duration_counts.values, edgecolor='black')
plt.title('Activity Duration (Days)')
plt.xlabel('Days')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.bar(frequency_counts.index, frequency_counts.values, edgecolor='black')
plt.title('Average Posting Frequency')
plt.xlabel('Hours Between Tweets')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'author_metrics_distribution_binned.png', dpi=300)
plt.show()

# Save ranking df
author_data.to_csv(DATASETS_DIR /'author_activity_ranking.csv')

# ===== Display Top 10 Most Active Users =====

top_10_table = author_data.nlargest(10, 'Score').reset_index()

# Convert author_id to string to avoid scientific notation
top_10_table['author_id'] = top_10_table['author_id'].astype(str)

# Convert duration to days
top_10_table['Activity Duration'] = top_10_table['Activity Duration'].dt.days

# Round and convert numeric columns to integers
top_10_table['Tweet Volume'] = top_10_table['Tweet Volume'].round().astype(int)
top_10_table['Activity Duration'] = top_10_table['Activity Duration'].round().astype(int)
top_10_table['Frequency in Hours'] = top_10_table['Frequency in Hours'].round().astype(int)
top_10_table['Score'] = top_10_table['Score'].round().astype(int)

# Select columns
display_cols = [
    'author_id',
    'Tweet Volume',
    'Activity Duration',
    'Frequency in Hours',
    'Score'
]

table_data = top_10_table[display_cols]

print('Top 10 Authors by Score (Activity): \n')
print(table_data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(display_cols))))

plt.title('Top 10 Most Active Users', pad=20)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'top_10_authors.png', dpi=300)
plt.show()