
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR = ROOT / "plots"


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Load datasets
df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')


# ===== Section 1: Burst Analysis ======

# Load precomputed results
burst_dates = pd.read_csv(DATASETS_DIR / 'bursts.csv', index_col='Date', parse_dates=['Date'])
burst_origins_per_author = pd.read_csv(OUTPUTS_DIR / 'burst_analysis' / 'burst_periods_per_author.csv')
avg_lag_times = pd.read_csv(OUTPUTS_DIR / 'burst_analysis' / 'avg_author_lag.csv')
author_lag_times = pd.read_csv(OUTPUTS_DIR / 'burst_analysis' / 'author_lag_times.csv')
ranking_df = pd.read_csv(OUTPUTS_DIR / 'burst_analysis' / 'burst_contribution_ranks.csv')
engagement_metrics = pd.read_csv(OUTPUTS_DIR / 'author_engagement_metrics.csv')


# === Part 1: Contributions to bursts ===

# Filter tweets that fall within burst dates
df['Date'] = df['created_at'].dt.date
burst_tweets = df[df['Date'].isin(burst_dates.index.date)]

# Tweets per (day, author)
tweets_per_author_day = burst_tweets.groupby(['Date', 'author_id']).size()

# Total tweets per burst day
total_tweets_day = burst_tweets.groupby('Date').size()

# % contribution per author per burst day
contribution_pct = (tweets_per_author_day / total_tweets_day * 100).reset_index(name='contribution_pct')

# Number of burst days each author appears in
author_burst_counts = contribution_pct.groupby('author_id')['Date'].nunique()

participated_in_threshold = 30

# Authors active in many bursts
consistent_authors = author_burst_counts[author_burst_counts >= participated_in_threshold].index

# Average contribution (%) across bursts
avg_contribution_pct = (
    contribution_pct[contribution_pct['author_id'].isin(consistent_authors)]
    .groupby('author_id')['contribution_pct']
    .mean()
    .sort_values(ascending=False)
)

print('Consistently active authors across burst periods')
print(f"Total authors appearing in at least {participated_in_threshold} bursts: {len(consistent_authors)}\n")

print('\nTop authors by average contribution during bursts (in %)\n')
print(avg_contribution_pct.head(10))

# Summary: participation + average contribution
burst_contributors = pd.DataFrame({
    'burst_count': author_burst_counts,
    'avg_contribution_pct': avg_contribution_pct
}).dropna().sort_values(by='burst_count', ascending=False)

print('\nAuthors consistently contributing across bursts: ')
print('(Reminder: "burst_count --> burst periods an author participated in'
      '\n"avg_contribution_pct" --> how much of the total tweet volume came from each author, per burst period)\n')
print(burst_contributors.head(10))
print('-' * 50)


# === Part 2: Find Catalysts ===

# Authors sorted by how early they tweet before bursts
early_authors = avg_lag_times.sort_values(by='first_tweet_lag_hours')

# Number of pre-burst windows each author appears in
author_preburst_counts = author_lag_times.groupby('author_id').size()

# Combine lag + consistency
catalyst_df = avg_lag_times.copy()
catalyst_df['burst_count'] = catalyst_df['author_id'].map(author_preburst_counts)

# Keep authors appearing in multiple pre-bursts
catalyst_df = catalyst_df[catalyst_df['burst_count'] >= 2]

# Prioritize earliest actors
catalyst_df = catalyst_df.sort_values(by='first_tweet_lag_hours')

print('\nPotential catalyst authors (close to burst starts and consistent)\n')
print(catalyst_df.head(10))
print('-' * 50)


# === Part 3: Retweet amplification during bursts ===

# Make author_id a column
engagement_metrics = engagement_metrics.reset_index()

# Keep only needed columns
engagement_for_bursts = engagement_metrics[['author_id', 'avg_retweets_per_tweet']]

# Retweets per tweet during pre-burst/burst windows
ranking_df['burst_retweets_per_tweet'] = (ranking_df['early_retweets'] / ranking_df['early_tweet_count'])

# Merge baseline engagement with burst behavior
retweet_comparison = ranking_df.merge(
    engagement_for_bursts,
    on='author_id',
    how='inner'
)

# Relative increase in retweet rate
retweet_comparison['retweet_boost'] = (
    retweet_comparison['burst_retweets_per_tweet'] / retweet_comparison['avg_retweets_per_tweet']
)

# Filter noise / unstable cases
retweet_comparison = retweet_comparison[
    (retweet_comparison['early_tweet_count'] >= 3) &
    (retweet_comparison['participated_in'] >= 2) &
    (retweet_comparison['avg_retweets_per_tweet'] > 0)
]

# Rank by amplification effect
top_propagators = retweet_comparison.sort_values(
    by='retweet_boost',
    ascending=False
)

print('\nAuthors with disproportionate retweet increase during bursts:\n')
print(top_propagators.head(10)[[
    'author_id',
    'avg_retweets_per_tweet',
    'burst_retweets_per_tweet',
    'retweet_boost',
    'early_tweet_count',
]])
print('-' * 50)


# === Part 4: Analyze posting frequency and timing patterns for authors filtered above ===

# Define the set of "important" authors, combining consistent contributors, catalysts and propagators
important_authors = set(burst_contributors.index) \
    .union(set(catalyst_df['author_id'])) \
    .union(set(top_propagators['author_id']))


print(f'\nTotal important authors analyzed in Part 4: {len(important_authors)}')

# Filter only important authors
burst_subset = burst_tweets[burst_tweets['author_id'].isin(important_authors)]

# Count tweets per author per burst day
tweets_per_author_per_day = (
    burst_subset
    .groupby(['Date', 'author_id'])
    .size()
    .reset_index(name='tweet_count')
)

# For each author, compute average tweets per burst (frequency)
avg_burst_frequency = (
    tweets_per_author_per_day
    .groupby('author_id')['tweet_count']
    .mean()
    .sort_values(ascending=False)
)

print('\nTop authors by average posting frequency during bursts (avg number of tweets per burst day):\n')
print(avg_burst_frequency.head(10))
print('-' * 50)


# === Part 5: Find and analyze possible coordination-between-users patterns in terms of tweeting timing ===

# Extract temporal features (hour of posting)
burst_subset['hour'] = burst_subset['created_at'].dt.hour

# Build distribution: rows: authors, columns: hours (0–23), entries: number of posts from author i in time j
author_time_distribution = (
    burst_subset
    .groupby(['author_id', 'hour'])
    .size()
    .unstack(fill_value=0)
)

# Normalize each row to sum to 1
author_time_distribution = author_time_distribution.div(
    author_time_distribution.sum(axis=1), axis=0
)


# Measure similarity between authors (temporal behavior)
similarity_matrix = cosine_similarity(author_time_distribution)

# Convert to DataFrame for readability
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=author_time_distribution.index,
    columns=author_time_distribution.index
)


# Identify highly similar author pairs
similarity_threshold = 0.9
coordinated_pairs = []

for i in range(len(similarity_df)):
    for j in range(i + 1, len(similarity_df)):
        if similarity_df.iloc[i, j] > similarity_threshold:
            coordinated_pairs.append((
                similarity_df.index[i],
                similarity_df.index[j],
                similarity_df.iloc[i, j]
            ))

print(f'\nHighly similar (potentially coordinated) author pairs (similarity > {similarity_threshold}):')
print(f'Total pairs found: {len(coordinated_pairs)}')

print('\nExamples:\n')
for pair in coordinated_pairs[:10]:
    print(f'Author {pair[0]} & Author {pair[1]}: similarity of {pair[2]:.2f}')


# Cluster authors based on temporal behavior
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

author_time_distribution['cluster'] = kmeans.fit_predict(author_time_distribution)

print('\nCluster distribution (authors grouped by posting time patterns):\n')
print(author_time_distribution['cluster'].value_counts())

# Interpret clusters
cluster_analysis = author_time_distribution.copy()
cluster_analysis['avg_burst_frequency'] = cluster_analysis.index.map(avg_burst_frequency)

print('\nSample of clustered authors with their burst activity: \n')
print(cluster_analysis.head(10))

# Look for a stronger coordination signal (same time, same burst)
co_tweeting = (
    burst_subset
    .groupby(['Date', 'hour', 'author_id'])
    .size()
    .reset_index()
)

# Count how often pairs co-appear in the same hour-window of the same burst
co_occurrence = {}

for (date, hour), group in co_tweeting.groupby(['Date', 'hour']):
    authors = list(group['author_id'])
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            pair = tuple(sorted((authors[i], authors[j])))
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

# Keep only frequently co-occurring pairs
coordinated_cooccurrence = {
    pair: count for pair, count in co_occurrence.items() if count >= 3
}

# Convert to DataFrame for sorting
coord_df = pd.DataFrame(
    list(coordinated_cooccurrence.items()),
    columns=['author_pair', 'co_occurrences']
)

# Sort descending by number of co-occurrences
coord_df = coord_df.sort_values(by='co_occurrences', ascending=False)

print('\nTop coordinated author pairs (most frequent co-tweeting first):\n')
print(coord_df.head(10))

# Flatten all author_ids from the coordinated pairs
all_authors_in_pairs = [author for pair in coord_df['author_pair'] for author in pair]

# Count how many times each author appears in these pairs
author_connectivity = Counter(all_authors_in_pairs)

# Convert to DataFrame for presentation
author_connectivity_df = pd.DataFrame.from_dict(
    author_connectivity, orient='index', columns=['num_coordinated_pairs']
)

# Sort (most “connected” authors first)
author_connectivity_df = author_connectivity_df.sort_values('num_coordinated_pairs', ascending=False)
print('\nAuthor connectivity (number of coordinated pairs each author is in):\n')
print(author_connectivity_df.head(10))
print('-' * 50)

print('-' * 100)


# ===== Section 2: Combine Results with Previous Analysis ======


# Load necessary for this section precomputed results
activity_ranking = pd.read_csv(OUTPUTS_DIR / 'author_activity_ranking.csv', index_col=0)
frequency_data = pd.read_csv(OUTPUTS_DIR / 'frequency_data.csv', index_col=0)


# Keep users that were found to show "coordination patterns" in at least ~20% of total burst days
coordination_threshold = 30

# Filter pairs meeting threshold
strong_pairs = coord_df[coord_df['co_occurrences'] >= coordination_threshold]

# Extract unique authors from these pairs
coordinated_authors = set()
for pair in strong_pairs['author_pair']:
    coordinated_authors.update(pair)

# Keep only highly connected authors among coordinated ones
highly_connected_authors_df = author_connectivity_df.loc[
    (author_connectivity_df.index.isin(coordinated_authors)) &
    (author_connectivity_df['num_coordinated_pairs'] >= 200)
]

# Extract author IDs from filtered DataFrame
highly_connected_author_ids = highly_connected_authors_df.index.tolist()


print(f'\nTotal authors showing strong coordination with many other users: {len(highly_connected_author_ids)}')
print('-' * 50)

# === Part 1: Combine with ranking results regarding contribution to burst initiations ===

# In the ranking df, find the rank of each author
ranking_df['rank'] = ranking_df.index + 1

# Filter highly connected authors and find their rank
coordinated_author_ranks = ranking_df.loc[
    ranking_df['author_id'].isin(highly_connected_author_ids),
    ['author_id', 'rank']
].sort_values(by='rank')

# Set author_id as index
coordinated_author_ranks.set_index('author_id', inplace=True)

# Rename rank column
coordinated_author_ranks.rename(columns={'rank': 'burst_contribution_rank'}, inplace=True)


# === Part 2: Combine with ranking results regarding total engagement metrics ===

# Sort engagement_metrics by Total Score descending
engagement_metrics_sorted = engagement_metrics.sort_values('Total Score', ascending=False).reset_index(drop=True)

# Compute the rank of each author in this df
engagement_metrics_sorted['total_engagement_rank'] = engagement_metrics_sorted.index + 1

# Filter only coordinated authors
engagement_ranks = engagement_metrics_sorted.loc[
    engagement_metrics_sorted['author_id'].isin(coordinated_authors),
    ['author_id', 'total_engagement_rank']
]

# Set author_id as index to match coordinated_ranks
engagement_ranks.set_index('author_id', inplace=True)

# Merge the two ranks into coordinated_ranks
coordinated_author_ranks = coordinated_author_ranks.merge(
    engagement_ranks,
    left_index=True,
    right_index=True,
    how='left'
)


# === Part 3: Combine with ranking results regarding author activity ===

# Sort activity rankings based on Score
activity_ranking_sorted = activity_ranking.sort_values('Score', ascending=False).reset_index()

# Compute activity rank
activity_ranking_sorted['activity_rank'] = activity_ranking_sorted.index + 1

# Filter only coordinated authors
activity_ranks = activity_ranking_sorted.loc[
    activity_ranking_sorted['author_id'].isin(coordinated_author_ranks.index),
    ['author_id', 'activity_rank']
]

# Set author_id as index to match coordinated_ranks
activity_ranks.set_index('author_id', inplace=True)

# Merge into coordinated_ranks
coordinated_author_ranks = coordinated_author_ranks.merge(
    activity_ranks,
    left_index=True,
    right_index=True,
    how='left'
)


# === Part 4: Combine with results regarding time-gaps-between-tweets analysis ===

# From frequency_data dataframe, keep only users' characterizations
frequency_char = frequency_data['author_type']

# Merge into coordinated_author_ranks
coordinated_author_ranks = coordinated_author_ranks.merge(
    frequency_char,
    left_index=True,
    right_index=True,
    how='left'
)

# ------------------------
print(coordinated_author_ranks.head(38))
