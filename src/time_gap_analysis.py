
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

# Sort the data based on author first, date second
df.sort_values(by=['author_id', 'created_at'], inplace=True)

# Group by author_id and get the time gaps between consecutive tweets for each author
df['time_since_last_tweet'] = df.groupby('author_id')['created_at'].diff()

# Get median and std of time gaps between consecutive tweets (uncomment print statements to show)
median_gap = df['time_since_last_tweet'].median()
std_gap = df['time_since_last_tweet'].std()
print('Median gap: ', median_gap)
print('std gap: ', std_gap)

# Create author summary (total tweets, average time gap between consecutive tweets, std of time gaps)
author_data = pd.DataFrame({
    'total_tweets': df.groupby('author_id').size(),
    'avg_gap_between_tweets': df.groupby('author_id')['time_since_last_tweet'].mean().dt.round('s'),
    'std': df.groupby('author_id')['time_since_last_tweet'].std().dt.round('s'),
})

# Decide which gaps between consecutive tweets are considered irregular (short or long)
df['is_short_gap'] = (df['time_since_last_tweet'] < pd.Timedelta(minutes=1))  # bot-like behavior
df['is_long_gap'] = (df['time_since_last_tweet'] > pd.Timedelta(days=20))     # "lurker" (passive user) behavior

# Count the number of irregular gaps between consecutive tweets, per author
author_data['short_count'] = df.groupby('author_id')['is_short_gap'].sum()
author_data['long_count'] = df.groupby('author_id')['is_long_gap'].sum()

# Calculate ratios (avoid division by zero for authors with 1 tweet)
tweet_pairs = author_data['total_tweets'] - 1
author_data['short_ratio'] = (author_data['short_count'] / tweet_pairs).fillna(0)
author_data['long_ratio'] = (author_data['long_count'] / tweet_pairs).fillna(0)

# Define irregular author types
author_data['author_type'] = 'normal'

# Bot-like: high proportion of short gaps with sufficient number of tweets to draw conclusions
author_data.loc[(author_data['short_ratio'] > 0.3) & (author_data['total_tweets']>5), 'author_type'] = 'bot_like'

# Lurker-like: high proportion of long gaps
author_data.loc[author_data['long_ratio'] > 0.5, 'author_type'] = 'lurker_like'

# Mixed: significant proportions of both
mixed = (author_data['short_ratio'] > 0.1) & (author_data['long_ratio'] > 0.2)
author_data.loc[mixed, 'author_type'] = 'mixed_irregular'

# One-time or low-activity authors (handle separately)
author_data.loc[author_data['total_tweets'] <= 3, 'author_type'] = 'insufficient_data'

print(author_data['author_type'].value_counts())

# Save dataset
author_data = author_data.reset_index()
author_data.to_csv(DATASETS_DIR / 'frequency_data.csv', index=False)