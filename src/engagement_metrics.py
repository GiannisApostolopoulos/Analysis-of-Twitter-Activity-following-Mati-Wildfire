from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
PLOTS_DIR = ROOT / "plots"
OUTPUTS_DIR = ROOT / "outputs"

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Load data
df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

# Get the total number of tweets per author
total_tweets = df.groupby('author_id').size()

# Group by author and get the metrics for each category per author
author_metrics = df.groupby('author_id').agg({
    'like_count': 'sum',
    'reply_count': 'sum',
    'retweet_count': 'sum'
})

# Add total tweets to the metrics df
author_metrics['total_tweets'] = total_tweets

# Average retweets per tweet
author_metrics['avg_retweets_per_tweet'] = (
    author_metrics['retweet_count'] / author_metrics['total_tweets']
)

# Total engagement score
author_metrics['Total Score'] = author_metrics[['like_count', 'reply_count', 'retweet_count']].sum(axis=1)

# Average engagement per tweet
author_metrics['Avg Engagement Per Tweet'] = (
    author_metrics['Total Score'] / author_metrics['total_tweets']
)

# === Visualization ===

# Histogram
plt.figure(figsize=(12, 6))
plt.hist(author_metrics['Avg Engagement Per Tweet'], bins=50, edgecolor='black')
plt.xlabel('Average Engagement per Tweet')
plt.ylabel('Number of Authors')
plt.title('Distribution of Average Engagement per Tweet')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'avg_engagement_per_tweet_distribution.png', dpi=300)
plt.close()

# Top 10 authors
top10_avg_engagement = author_metrics.nlargest(10, 'Avg Engagement Per Tweet')

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

table = ax.table(
    cellText=top10_avg_engagement.reset_index().values,
    colLabels=top10_avg_engagement.reset_index().columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 2)

plt.savefig(PLOTS_DIR / 'top10_avg_engagement_table.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
author_metrics.to_csv(OUTPUTS_DIR / 'author_engagement_metrics.csv')