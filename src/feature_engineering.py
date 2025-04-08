import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load data function
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

# TF-IDF Function
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Tf-Idf to the data and return transformed train and test DataFrames."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)  # ✅ correct usage

        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('TF-IDF applied and data transformed successfully.')
        return train_df, test_df

    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

# Save function
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

# Main driver function
def main():
    try:
        max_features = 50

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join('./data', 'processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('./data', 'processed', 'test_tfidf.csv'))

        logger.info('Feature engineering process completed successfully.')

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f'Error: {e}')

#  Important: __name__ check should be outside main()
if __name__ == '__main__':
    main()



