
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"

PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Load datasets
df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

# Create a new dataframe with one row per author
author_data = pd.DataFrame({
    'total_tweets': df.groupby('author_id').size(),
    'retweet_count': df.groupby('author_id')['Retweet'].sum()
})

print(f'Total authors in the dataset: {len(author_data)}')

# Percentage of retweets per author
author_data['retweet_perc'] = author_data['retweet_count'] / author_data['total_tweets']

# Percentage of original tweets
author_data['original_perc'] = 1 - author_data['retweet_perc']

# Categories based on posting behavior
author_data['retweet_category'] = 'mixed'
author_data.loc[author_data['retweet_perc'] == 0, 'retweet_category'] = 'original_only'
author_data.loc[author_data['original_perc'] == 0, 'retweet_category'] = 'retweets_only'

# Number of original tweets each author has
author_data['original_count'] = author_data['total_tweets'] - author_data['retweet_count']

# Keep only authors with at least 1 original tweet for ratio calculations
authors_with_original = author_data[author_data['original_count'] > 0].copy()

# Ratio of retweets to original tweets
authors_with_original['rt_og_ratio'] = authors_with_original['retweet_count'] / authors_with_original['original_count']


print('\nUser categorization summary:')

# Category counts
original_only = author_data[author_data['retweet_category'] == 'original_only']
retweets_only = author_data[author_data['retweet_category'] == 'retweets_only']
mixed = author_data[author_data['retweet_category'] == 'mixed']

print(f'\nOriginal-only users: {len(original_only)}')
print(f'Retweet-only users: {len(retweets_only)}')
print(f'Mixed users: {len(mixed)}')
print('-' * 50)

print('\nTop 10 mixed users by retweet/original ratio:')
print()
top_mixed = authors_with_original.sort_values('rt_og_ratio', ascending=False).head(10)
print(top_mixed[['total_tweets', 'original_count', 'retweet_count', 'rt_og_ratio']].to_string())
print('-' * 50)

print('\nTop 10 original-only users (by total tweets): ')
print()
top_original = original_only.sort_values('total_tweets', ascending=False).head(10)
print(top_original[['total_tweets', 'original_count']].to_string())
print('-' * 50)


print('\nTop 10 retweet-only users (by total tweets):')
print()
top_retweets = retweets_only.sort_values('total_tweets', ascending=False).head(10)
print(top_retweets[['total_tweets', 'retweet_count']].to_string())
print('-' * 50)

# Top 10 mixed users by original-to-retweet ratio
mixed_with_retweets = authors_with_original[authors_with_original['retweet_count'] > 0].copy()
mixed_with_retweets['og_rt_ratio'] = mixed_with_retweets['original_count'] / mixed_with_retweets['retweet_count']

print('\nTop 10 mixed users by original-to-retweet ratio:')
top_mixed_og_rt = mixed_with_retweets.sort_values('og_rt_ratio', ascending=False).head(10)
print(top_mixed_og_rt[['total_tweets', 'original_count', 'retweet_count', 'og_rt_ratio']].to_string())
print('-' * 50)


print('\nMixed users - Basic statistics:')
print(f'\nAverage retweet/original ratio: {mixed["retweet_count"].sum() / mixed["original_count"].sum():.4f}')
print(f'Median retweet/original ratio: {authors_with_original["rt_og_ratio"].median():.4f}')
print(f'Average total tweets per user: {mixed["total_tweets"].mean():.2f}')
print(f'Average original tweets per user: {mixed["original_count"].mean():.2f}')
print(f'Average retweets per user: {mixed["retweet_count"].mean():.2f}')


# Bar plot showing distribution of users across categories
plt.figure(figsize=(8, 6))
category_counts = [len(original_only), len(retweets_only), len(mixed)]
category_labels = ['Original Only', 'Retweet Only', 'Mixed']
colors = ['#2ecc71', '#e74c3c', '#3498db']

bars = plt.bar(category_labels, category_counts, color=colors, alpha=0.7)
plt.title('Distribution of Users by Posting Behavior', fontsize=14, fontweight='bold')
plt.xlabel('User Category')
plt.ylabel('Number of Users')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'user_category_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# Histogram of retweet percentage distribution
plt.figure(figsize=(10, 6))
plt.hist(author_data['retweet_perc'], bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title('Distribution of Retweet Percentage Across All Users', fontsize=14, fontweight='bold')
plt.xlabel('Retweet Percentage')
plt.ylabel('Number of Authors')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'retweet_percentage_distribution.png', dpi=300, bbox_inches='tight')
plt.show()