# In this file, we implement methods to filter out irrelevant tweets, ensuring that the subsequent
# analysis is as accurate and reliable as possible. To develop these mechanisms, we relied on
# the example code provided for the Tempi dataset. Furthermore, we clean the dataset
# by removing columns that contain no useful information.


import pandas as pd
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
OUTPUTS_DIR = ROOT / "outputs"

DATASETS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def preprocess_text(text):
    """
    This function takes a text series and cleans it
    :param text: pd Series
    :return: cleaned text column
    """
    # Helper function to remove punctuation. This will be useful in the filtering stage later
    def remove_accents(string):
        import unicodedata
        string= str(string)
        # Normalize text to NFD form (decomposes letters + accents)
        string = unicodedata.normalize('NFD', string)
        # Keep only characters that are not combining marks (removes accents)
        string = ''.join(c for c in string if not unicodedata.combining(c))
        return string

    # Convert text to lowercase
    text = text.str.lower()
    # Remove mentions
    text = text.str.replace(r"@\w+", "", regex=True)
    # Replace underscores with spaces
    text = text.str.replace("_", " ", regex=False)
    # Remove retweet indicators
    text = text.str.replace(r"\b(rt)\b", "", regex=True)
    # Remove links
    text = text.str.replace(r"http\S+|www\S+", "", regex=True)
    # Remove hashtags
    text = text.str.replace(r"#", "", regex=True)  # Matches the '#' character
    # Remove accents
    text = text.apply(remove_accents)
    # Remove redundant spaces
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    # Remove consecutive punctuation marks (e.g., !!!, ??)
    text = text.str.replace(r"[.!?;&]{1,}", "", regex=True)
    # Remove single full stops, commas, colons, and hyphens
    text = text.str.replace(r"[.,:\-]", "", regex=True)  # Matches ., :, -, or ,
    # Remove words with less than 4 characters
    text = text.str.replace(r'\b\w{1,4}\b', '', regex=True)
    # Remove ellipsis-like characters (…)
    text = text.str.replace(r"…", "", regex=True)  # Matches the Unicode ellipsis character '…'
    text = text.str.replace(
        r"[\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        r"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        r"\U00002600-\U000026FF"  # Miscellaneous Symbols
        r"\U00002700-\U000027BF"  # Dingbats
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        r"]+",
        "",
        regex=True
    )  # Matches Unicode ranges for various emoji and symbols

    return text


def drop_extreme_columns(dataframe):
    """
    Function that will remove from the DataFrame columns that contribute no useful info.
    These are columns that contain only one distinct value,
    as many distinct values as the number of entries in the DataFrame, and columns that contain only NaN's.
    However, we will keep the column "tweet_id" in case we wish to look up original text in the sequel.
    :param dataframe: pd dataframe
    :return: Cleaned DataFrame
    """
    # create a copy to avoid modifications on the df while iterating
    dataframe_clean = dataframe.copy()

    for col in dataframe.columns:
        # If column is the tweet_id, do not drop it
        if col == 'tweet_id':
            continue
        # Find the number of distinct entries in each column
        n_unique = dataframe[col].nunique()
        if n_unique == 1 or n_unique == len(dataframe) or dataframe[col].isna().all():
            dataframe_clean.drop(columns=col, inplace=True)

    return dataframe_clean


def remove_stopwords(text):
    """
    Function that removes stopwords from a text. This makes the filtering-with-keywords process that follows more
    accurate.
    :param text: pd Series object
    :return: cleaned text
    """

    stopwords = {
        "επειδη", "αυτοι", "γιατι", "εχουμε", "εχει", "είναι", "ειμαστε",
        "στους", "στον", "τους", "παντα", "καμια", "ποιος", "καποιος",
        "εχουν", "ειχαν", "εκανε", "οποιος", "ειχε", "κανουν", "εχουν",
        "στην", "εναν", "ακομα", "ακομη", "αλλα", "αλλη", "αλλο", "αλλος",
        "αμεσα", "αλλιως", "αυτον", "αυτος", "καποια", "καποιοι", "εκτος",
        "εκεινος", "εκεινοι", "ηταν", "τετοια", "τετοιο",
    }

    # Helper function that actually removes stopwords from string
    def clean_text(txt):
        words = txt.split()
        filtered = [word for word in words if word not in stopwords]
        return ' '.join(filtered)

    # Apply the function to our series
    return text.apply(clean_text)


def expand_keywords(text, seed_keywords, iters=2, n=40):
    """
    Function that expands a list of seed keywords, by finding the words that most frequently
    accompany these seed keywords and appending them to the list.
    Before applying this function, we first need to clean the text column.
    :param text: pd Series object
    :param seed_keywords: list of initial keywords
    :param iters: repetitions of the process
    :param n: number of keywords to be appended in the initial list in each repetition
    :return: the final list of keywords
    """

    # Create a copy of the initial list
    expanded_keywords = [w.lower() for w in seed_keywords]

    # Drop cells that contain no text
    text = text.dropna()

    # Helper function
    def contains_keywords(txt):
        words = txt.split()
        return any(word in words for word in expanded_keywords)

    # Repeat "iters" times
    for _ in range(iters):

        # Find the tweets that contain any of the words in the current list
        mask = text.apply(contains_keywords)
        filtered_tweets = text[mask]

        # Get all the words from those tweets
        all_words = ' '.join(filtered_tweets).split()

        # Count the occurrences of each word
        count = Counter(all_words)

        # We don't want to grab words that we already have in our keywords list, so remove those from the list
        # that we will filter
        for word in expanded_keywords:
            if word in count:
                count.pop(word)

        # Get the n most common words
        most_common = count.most_common(n)
        # Append them in our keywords list
        expanded_keywords.extend([word for word, count in most_common])


        expanded_keywords = list(set(expanded_keywords))

    return expanded_keywords


def filter_tweets(text, seed_keywords, iters=2, n=40):
    """
    Function that filters tweets based on keywords.
    :param text: pd Series object
    :param seed_keywords: list of initial keywords
    :param iters: repetitions of the process
    :param n: number of keywords to be appended in the initial list in each repetition
    :return: (filter, list of final keywords)
    """
    # Expand the list of keywords by applying the function defined above
    expanded_keywords = expand_keywords(text, seed_keywords, iters=iters, n=n)

    # Helper function to check if a tweet contains any of the keywords
    def contains_keywords(txt):
        words = txt.split()
        return any(word in words for word in expanded_keywords)

    # Create the filter to be applied in the series
    mask = text.apply(contains_keywords)

    return mask, expanded_keywords


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)

    # Load the dataset
    df = pd.read_csv(DATASETS_DIR / 'mati.csv', header=None)

    # Rename the columns of the DataFrame, extracting the names from the example code
    df.columns = ["author_id", "created_at", "geo", "tweet_id", "lang", "like_count",
                  "quote_count", "reply_count", "retweet_count", "source", "text"]

    # Remove columns that contribute no useful info for our analysis
    df.drop(columns=['lang', 'geo', 'quote_count'], inplace=True)

    # Create a new column that indicates whether an entry is a retweet or not
    df['Retweet'] = df['text'].str.startswith('RT')

    # ============ Clean the dataset ============
    df = drop_extreme_columns(df)
    df['text'] = preprocess_text(df['text'])
    df['text'] = remove_stopwords(df['text'])

    seed_keywords = [
        "φωτια", "ματι", "πυρκαγια", "νεα δημοκρατια", "συριζα", "τσιπρας",
        "μητσοτακης", "κουλης", "disaster", "νεκροι", "νεκρους", "πυρκαγιες",
        "φωτιες", "lives", "εκλογες", "συριζαιοι", "ζαιοι", "συμμορια", "μαφια",
         "ομαδα", "αληθειας", "δουρου", "ρουφιανοι", "πυροσβεστικη", "προαγωγη",
         "προαγωγες", "θυματα", "κυβερνηση", "ραφηνα", "τραγωδια",
         "κινετα", "ραφηνα", "δικη", "τεμπη", "τρενα", "κατηγορουμενοι", "στελεχη",
         "ξεχασει", "μνημης", "εφημεριδα", "απατεωνες", "ανθρωπινες", "απατεωνες",
         "τυμβωρυχια", "αποζημιωση", "ημερα", "πραγματογνωμονας", "δικαιοσυνη",
         "εκκενωση", "καταθεση", "συγκαλυψη", "μαρτυρια", "πρωτοσελιδο",
         "μαραθωνα", "ανακριτης", "πυροπληκτους", "πυροπληκτοι", "καμμενος",
         "τσουβαλας", "τσουβαλα", "πορισμα", "δικογραφια", "πυρκαγιας", "συγγενεις",
         "πυρκαγια", "ανεμος", "αερας", "αερα", "προστασια", "φλογες", "αγνοειται",
         "περιθαλψη", "τροφιμα", "αναγκης", "αδεσποτα", "χαθηκε", "φαρμακα",
         "τοσκας", "τοσκα", "νομοσχεδιο", "κατασβεση", "περιουσιες", "διαχειριση",
         "πυρκαγιων", "εκατομβη", "παραιτηση", "πυροπληκτοι", "μποφορ", "συγγενων",
         "κακουργημα", "επιτελικο", "μνημη", "ξεχνω", "ξεχαστει", "μαρτυριες", "σιδηροδρομος",
         "επιχειρησιακο", "τροφιμων", "σοροι", "families", "γραμμη βοηθειας", "καηκε",
         "αναζητουνται", "λιστα", "αιμοδοσια", "victims", "αυθαιρετα", "αναγκη", "νοσηλεύονται",
         "ψινακης", "υποκινουμενος", "διασωση", "πτωματα",
    ]



    mask, expanded_keywords = filter_tweets(df['text'], seed_keywords, iters=2, n=40)
    df_filtered = df[mask].copy()

    # Save the cleaned dataset
    df_filtered.to_csv(DATASETS_DIR / 'mati_clean.csv', index=False, encoding='utf-8-sig')

    # Tweets that didn't pass the filtering stage
    df_rejected = df[~mask].copy()
    # Save the rejected tweets in a separate csv file for inspection and verification of the filterin process
    df_rejected.to_csv(DATASETS_DIR / 'rejected.csv', index=False, encoding='utf-8-sig')

    print('Dataset was cleaned successfully and saved as "mati_clean.csv"')
    print(f'Entries kept: {len(df_filtered)}')
    print(f'Entries rejected: {len(df_rejected)}')
    print(f"\nTotal authors in the dataset: {df_filtered['author_id'].nunique()}")
    df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])
    print(f"Dataset entries spread across {df_filtered['created_at'].dt.date.min()} to {df_filtered['created_at'].dt.date.max()}")

