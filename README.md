# Twitter Data Analysis
## Analysis of Social Media Activity Regarding the 2018 Mati Wildfire


### Project Overview

This project analyzes Twitter activity surrounding the Mati wildfire in Greece. The goal is to understand tweet volume 
patterns, user activity, and "burst" events where tweet activity spiked. The analysis includes:

* Filtering and cleaning dataset
* User categorization based on various metrics
* Temporal analysis of tweet activity
* Identification of coordinated activity boosting engagement
* Visualization of patterns and trends

Methodological choices, assumptions, and limitations are explicitly discussed to ensure transparency and reproducibility.

All code, outputs, and figures are reproducible using the scripts in the src/ folder.

### Dataset

The Twitter dataset analyzed in this project covers tweets related to the 2018 Mati wildfire. 
It is publicly available and can be downloaded via the script provided in `src/download_dataset.py`. 

Original source / download link: [Download Dataset]( https://www.dropbox.com/scl/fi/y8wvktb5lefnozak13aru/mati.csv?rlkey=hl3f7wpwe6ruadgx4v0cecya0&st=uxf341rz&dl=0)

### Setup & Installation

1. Clone repository
   *     git clone <repository-url>
   *     cd <repository-folder>
2. Install required python packages
   *     pip install -r requirements.txt


### Usage

1. Download dataset
   *     python src/download_dataset.py
2. Run analysis scripts in order:
   *     python src/filtering.py
   *     python src/burst_detection.py
   *     python src/author_activity.py
   *     python src/engagement_metrics.py
   *     python src/retweet_dependency.py
   *     python src/time_analysis.py
   *     python src/time_gap_analysis.py
   *     python src/weekly_patterns.py
   *     python src/burst_origin_analysis.py
   *     python src/influential_authors.py

### Documentation

Refer to the docs/report.pdf for a detailed discussion of the methodology, results, and insights.

### License

This project is for research and educational purposes. The dataset analyzed is publicly available and was collected 
during a period when Twitter allowed research-oriented data mining. No proprietary or private data is included in this 
repository, and all code and analyses can be reproduced using the provided scripts.