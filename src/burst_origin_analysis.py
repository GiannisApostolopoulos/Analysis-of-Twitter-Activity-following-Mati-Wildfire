
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
PLOTS_DIR = ROOT / "plots"
OUTPUTS_DIR = ROOT / "outputs" / "burst_analysis"

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Load csv's
df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')
bursts = pd.read_csv(DATASETS_DIR / 'bursts.csv', index_col='Date', parse_dates=['Date'])

# From the original df, get the time of each tweet
df['Hour of Day'] = df['created_at'].dt.hour

# Get the date part of the timestamps in "created_at" column, to match the indices of "bursts" DataFrame
df['Date'] = df['created_at'].dt.date

# Identify the tweets posted in "first burst days" in our whole dataset
burst_tweets = df[df['Date'].isin(bursts.index.date)]

# For comparison, get the tweets that were posted the day before each "first burst day"
previous_days = [day - timedelta(days=1) for day in bursts.index.date]
previous_day_tweets = df[df['Date'].isin(previous_days)]

burst_days_count = burst_tweets['Date'].nunique()
prev_days_count = previous_day_tweets['Date'].nunique()

print(f"Found {len(burst_tweets):,} tweets from {burst_days_count} burst days")
print(f"Found {len(previous_day_tweets):,} tweets from {prev_days_count} days before bursts")
print('-' * 50)

total = (
    previous_day_tweets['author_id'].nunique() +
    burst_tweets[burst_tweets['Hour of Day'] < 8]['author_id'].nunique()
)
print(f"Found {total} unique users tweeting in a pre-burst window.")
del total

# === Burst days analysis ===

# Group by hour and get total tweets per hour across burst days
tweets_per_hour_burst = burst_tweets.groupby('Hour of Day').size()
# Get the number of first burst days
total_fburst_days = burst_tweets['Date'].nunique()
# Calculate average per hour for first burst days
avg_tweets_per_hour_burst = tweets_per_hour_burst / total_fburst_days

# === Previous days analysis ===

# Group by hour and get total tweets per hour across days before bursts
tweets_per_hour_prev = previous_day_tweets.groupby('Hour of Day').size()
# Get the number of previous days. This number is not the same as "total_fburst_days" because some preceding dates
# are missing from the dataset
total_prev_days = previous_day_tweets['Date'].nunique()
# Calculate average per hour for days before bursts
avg_tweets_per_hour_prev = tweets_per_hour_prev / total_prev_days

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'Hour': range(24),
    'Burst Days': [avg_tweets_per_hour_burst.get(h, 0) for h in range(24)],
    'Day Before': [avg_tweets_per_hour_prev.get(h, 0) for h in range(24)]
})

# Visualize

plt.figure(figsize=(12, 6))
plt.plot(range(24), comparison_df['Burst Days'], 'r-', label='Burst Days', linewidth=2)
plt.plot(range(24), comparison_df['Day Before'], 'b--', label='Day Before Burst', linewidth=2)
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Tweets')
plt.title('Average Tweet Volume by Hour: Burst Days vs. Days Before')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'tweet_volume_burst_and_preburst_days.png', dpi=300, bbox_inches='tight')
plt.show()

print('-' * 50)


# Get the tweets that are flagged as "early", for each burst period
early_burst_tweets = pd.concat([
    burst_tweets[burst_tweets['Hour of Day'].isin(range(8))],
    previous_day_tweets[previous_day_tweets['Hour of Day'].isin(range(18, 24))]
])

# Get the authors of those tweets
early_burst_users = early_burst_tweets.groupby('author_id')
# Count the number of early tweets per user
total_early_volume = early_burst_users.size()
# Get the unique author IDs of "early authors"
early_author_ids = early_burst_tweets['author_id'].unique()
# Get the number of total tweets per user
total_tweet_volume = df[df['author_id'].isin(early_author_ids)].groupby('author_id').size()
# For each of these users, calculate the percentage of their posts that were posted during an early burst period
contribution_percentage = (total_early_volume / total_tweet_volume * 100)


# === Lag time calculation ===

# Get the set of burst dates for easy lookup
burst_dates = set(bursts.index.date)


# Calculate time difference to onset (8:00 AM) for each tweet
def calculate_lag_to_onset(row):
    # Handle timezone issues
    tweet_time = row['created_at']
    if hasattr(tweet_time, 'tz') and tweet_time.tz is not None:
        tweet_time = tweet_time.tz_localize(None)

    # Create onset datetime for the relevant day (8:00 AM)
    if row['Date'] in burst_dates:
        # It's a burst day - onset is 8:00 AM on the same day
        onset = pd.Timestamp.combine(tweet_time.date(), pd.Timestamp('08:00:00').time())
    else:
        # It's a previous day - onset is 8:00 AM on the next day (burst day)
        next_day = tweet_time.date() + timedelta(days=1)
        onset = pd.Timestamp.combine(next_day, pd.Timestamp('08:00:00').time())

    # Calculate lag in hours (positive means before onset)
    return (onset - tweet_time).total_seconds() / 3600


# Apply to create new column
early_burst_tweets['hours_before_onset'] = early_burst_tweets.apply(calculate_lag_to_onset, axis=1)

# Create a mapping from date to its corresponding burst date. For "first burst days", the burst date is the date itself.
# For previous days, the burst date is the next day.

def get_burst_date(row):
    if row['Date'] in burst_dates:
        return row['Date']
    else:
        # It's a previous day, so burst date is the next day
        return row['Date'] + timedelta(days=1)

# Add burst_date column
early_burst_tweets['burst_date'] = early_burst_tweets.apply(get_burst_date, axis=1)


# For each author and burst day, find their first tweet (maximum hours before onset)
author_first_tweet_lag = early_burst_tweets.groupby(['author_id', 'burst_date'])['hours_before_onset'].max().reset_index()

# Rename for clarity
author_first_tweet_lag = author_first_tweet_lag.rename(columns={'hours_before_onset': 'first_tweet_lag_hours'})

# Get the average lag per author across all bursts
avg_author_lag = author_first_tweet_lag.groupby('author_id')['first_tweet_lag_hours'].mean().sort_values()

print('\nOverall statistics across all author-burst combinations:')
all_lags = author_first_tweet_lag['first_tweet_lag_hours']
print(f'\nAverage lag time: {all_lags.mean():.2f} hours')
print(f'Median lag time: {all_lags.median():.2f} hours')
print(f'Standard deviation: {all_lags.std():.2f} hours')
print(f'Minimum lag: {all_lags.min():.2f} hours')
print(f'Maximum lag: {all_lags.max():.2f} hours')

# Percentiles
print("\nPercentiles:")
for percentile in [10, 25, 50, 75, 90, 95]:
    print(f"{percentile}th percentile: {all_lags.quantile(percentile/100):.2f} hours")


print('-' * 50)


# === Ranking System ===

# Total engagement per user
total_early_engagement = early_burst_users[['like_count', 'reply_count']].sum().sum(axis=1)
# Total retweets per user
total_early_retweets = early_burst_users['retweet_count'].sum()

# Get the number of unique burst periods each author participated in
burst_periods_per_author = early_burst_tweets.groupby('author_id')['burst_date'].nunique().reset_index()
# Rename columns for clarity
burst_periods_per_author.columns = ['author_id', 'participated_in']

# Combine all metrics into a single DataFrame
ranking_df = pd.DataFrame({
    'author_id': total_early_volume.index,
    'early_tweet_count': total_early_volume.values,
    'early_engagement': total_early_engagement.values,
    'early_retweets': total_early_retweets.values,
    'pct_contribution': contribution_percentage.loc[total_early_volume.index].values
})

# Merge with burst periods
ranking_df = ranking_df.merge(burst_periods_per_author, on='author_id', how='left')

# Create weight vector
weights = {
    'early_tweet_count': 0.35,
    'early_retweets': 0.35,
    'early_engagement': 0.15,
    'pct_contribution': 0.15
}

# Calculate score per user
ranking_df['raw_score'] = (
    ranking_df['early_tweet_count'] * weights['early_tweet_count'] +
    ranking_df['early_retweets'] * weights['early_retweets'] +
    ranking_df['early_engagement'] * weights['early_engagement'] +
    ranking_df['pct_contribution'] * weights['pct_contribution']
)

# Sort by score
ranking_df = ranking_df.sort_values('raw_score', ascending=False).reset_index(drop=True)

# Get top 30
top_30 = ranking_df.head(30).copy()

print('\nTop 30 users by burst contribution (raw scores)')
print()
print(top_30[['author_id', 'early_tweet_count', 'early_engagement',
              'early_retweets', 'pct_contribution', 'raw_score', 'participated_in']].to_string(index=False))
print('-' * 50)

# Calculate and print averages for all users
print('\nAverages for all users (across all burst periods):')
print(f'\nAverage early tweet count: {ranking_df['early_tweet_count'].mean():.2f}')
print(f'Average early engagement: {ranking_df['early_engagement'].mean():.2f}')
print(f'Average retweet count "early tweets" received: {ranking_df['early_retweets'].mean():.2f}')
print(f'Average percentage contribution (early tweets / total tweets): {ranking_df['pct_contribution'].mean():.2f}%')
print(f'Average raw score: {ranking_df['raw_score'].mean():.2f}')
print(f'Average burst periods participated in: {ranking_df['participated_in'].mean():.2f}')


# === Save results ===

# Save the main ranking dataframe
ranking_df.to_csv(OUTPUTS_DIR / 'burst_contribution_ranks.csv', index=False)

# Save author lag times (for identifying catalysts)
author_first_tweet_lag.to_csv(OUTPUTS_DIR / 'author_lag_times.csv', index=False)

# Save average lag per author
avg_author_lag = avg_author_lag.reset_index()
avg_author_lag.to_csv(r'datasets/burst_analysis/avg_author_lag.csv', index=False)

# Save the raw early burst tweets with all metadata (for detailed timeline analysis)
early_burst_tweets.to_csv(OUTPUTS_DIR / 'early_burst_tweets.csv', index=False)

# Save the burst periods per author
burst_periods_per_author.to_csv(OUTPUTS_DIR / 'burst_periods_per_author.csv', index=False)

# Save the percentage contribution data
contribution_percentage.to_csv(OUTPUTS_DIR / 'percentage_contribution.csv', index=False)