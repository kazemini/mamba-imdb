import html
import os
import re

import kagglehub
import pandas as pd


class IMDBDataset:
    def __init__(self):
        self.dataset_id = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        self.dataset_path = self.download_dataset()
        
        self.df = self.load_dataset()
        self.df = self.preprocess_dataset()

    def download_dataset(self) -> str:
        """
        Downloads the IMDB dataset from Kaggle.
        Returns:
            str: The path to the downloaded dataset.
        """

        path = kagglehub.dataset_download(self.dataset_id)
        print("Dataset downloaded to:", path)
        return path

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the IMDB dataset from the downloaded CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the IMDB dataset.
        """
        csv_path = os.path.join(self.dataset_path, "IMDB Dataset.csv")
        return pd.read_csv(csv_path)

    def clean_text(self, text) -> str:
        """
        Cleans the input text by removing HTML tags, URLs, and extra whitespace.
        Returns:
            str: The cleaned text.
        """
        if pd.isna(text):
            return ""

        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_dataset(self) -> pd.DataFrame:
        """
        Preprocesses the IMDB dataset by cleaning the text and encoding the sentiment labels.
        Returns:
            pd.DataFrame: A DataFrame containing the preprocessed IMDB dataset.
        """
        self.df = self.df.dropna()
        self.df['review'] = self.df['review'].map(self.clean_text)
        self.df['sentiment'] = self.df['sentiment'].map({'positive': 1, 'negative': 0})
        return self.df