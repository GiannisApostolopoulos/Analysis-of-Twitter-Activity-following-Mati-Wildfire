
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

df = pd.read_csv(DATASETS_DIR / 'mati_clean.csv', parse_dates=['created_at'])
df_original = pd.read_csv(DATASETS_DIR / 'mati.csv')

# Extract day of week and week of year each tweet was posted
df['day_of_week'] = df['created_at'].dt.day_name()
df['week_of_year'] = df['created_at'].dt.isocalendar().week

# Get the total count of tweets per week of year, across dataset timeline
tweets_per_week_of_year = df.groupby('week_of_year').size()



# Create a pivot table: weeks as rows, days as columns, with tweet counts
# First, create a day of week order (Monday to Sunday)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create the pivot table
heatmap_data = pd.crosstab(
    index=df['week_of_year'],
    columns=df['day_of_week'],
    values=df['created_at'],  # any column will do, we just need counts
    aggfunc='count',
    normalize=False  # set to True if you want percentages instead of raw counts
).reindex(columns=day_order)

# Fill any missing values with 0
heatmap_data = heatmap_data.fillna(0)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    annot=True,  # shows the numbers in cells
    fmt='.0f',   # format as integers
    cmap='YlOrRd',  # yellow to red colormap (good for intensity)
    cbar_kws={'label': 'Number of Tweets'}
)

plt.title('Tweet Volume by Week of Year and Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Week of Year', fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'weekly_activity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Split into two periods:
# First, the crisis period, the one that followed the fire
# Second, the normal period, where the shock of the incident has faded and users return to regular posting patterns
crisis_end_date = '2018-08-31'  # drawn from burst_detection
df['period'] = 'normal'
df.loc[df['created_at'] < crisis_end_date, 'period'] = 'crisis'

# Analyze separately
engagement_by_day_crisis = df[df['period'] == 'crisis'].groupby('day_of_week').size()
engagement_by_day_normal = df[df['period'] == 'normal'].groupby('day_of_week').size()

# Compare patterns
print("Crisis period pattern:")
print(engagement_by_day_crisis.sort_values(ascending=False))
print("\nNormal period pattern:")
print(engagement_by_day_normal.sort_values(ascending=False))