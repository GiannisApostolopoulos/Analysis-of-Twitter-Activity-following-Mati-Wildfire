
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"

PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

# Extract hour from timestamps
df['Hour of Day'] = df['created_at'].dt.hour

# Group by hour and get total tweets per hour across dataset timeline
tweets_per_hour = df.groupby('Hour of Day').size()
# Calculate the length of the dataset timeline
total_days = (df['created_at'].max() - df['created_at'].min()).days
# Get the average tweets posted per hour
avg_tweets_per_hour = tweets_per_hour / total_days

# Find peak and trough
peak_hour = avg_tweets_per_hour.idxmax()
peak_value = avg_tweets_per_hour.max()
lowest_hour = avg_tweets_per_hour.idxmin()
lowest_value = avg_tweets_per_hour.min()

print(f"Peak activity: {peak_value:.2f} tweets at {peak_hour}:00")
print(f"Lowest activity: {lowest_value:.2f} tweets at {lowest_hour}:00")
print(f"Activity range: {peak_value / lowest_value:.2f} difference")


# Create bar plot of average tweets per hour
plt.figure(figsize=(12, 6))
avg_tweets_per_hour.plot(kind='bar')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Tweets')
plt.title('Average Tweet Activity by Hour of Day')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'avg_tweets_by_hour.png', dpi=300)
plt.show()