import ssl
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Handle SSL error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data():
    # Load the data
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')

    # Merge datasets and create labels
    true_df['label'] = 1
    fake_df['label'] = 0
    df = pd.concat([true_df, fake_df])

    # Convert all text to lowercase
    df['text'] = df['text'].str.lower()

    # Remove punctuation
    df['text'] = df['text'].str.replace('[^\w\s]','')

    # Tokenize text
    df['text'] = df['text'].apply(nltk.word_tokenize)

    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stop_words])

    # Join tokens back into string
    df['text'] = df['text'].apply(' '.join)

    # Vectorize text
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text']).toarray()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
