"""
This script implements a simple text classifier using logistic regression from sklearn.

The classifier uses the following features:
* TF-IDF: Term Frequency-Inverse Document Frequency
* Average number of characters per word
* Average number of words per sentence
* Number of sentences per document
* Additional features: Unique word count, stopword count, punctuation count, word diversity, average word length, capitalization ratio

Key Modifications and Comments:
1. Organized and cleaned up the code for better readability and functionality.
2. Introduced a single method `compute_features` in the `FeatureComputer` class with an `include_additional` flag to optionally compute extended features.
3. Implemented the computation of TF-IDF manually without relying on external libraries like `sklearn`. The formula used is:
    - Term Frequency (TF): `TF(word) = (Number of occurrences of word in document) / (Total words in document)`
    - Inverse Document Frequency (IDF): `IDF(word) = log((Number of documents) / (1 + Number of documents containing word)) + 1`
    - TF-IDF: `TF-IDF(word) = TF(word) * IDF(word)`
Why this way:
Term Frequency (TF) captures how often a word appears in a document, normalized to avoid bias toward longer documents. Inverse Document Frequency (IDF) reduces
the weight of common words by penalizing terms frequent across all documents. Multiplying TF and IDF emphasizes terms that are both frequent in a document and 
rare in the corpus, improving discriminative power.
4. Added comments throughout the code explaining functionality and changes for clarity.
5. Ensured the code runs on Python 3.6 or higher and outputs results as required.
6. Predictions are saved to CSV files (`simple_model_predictions.csv` and `best_model_predictions.csv`).
"""

import pandas as pd
import numpy as np
import string
from math import log
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Downloading necessary NLTK resources
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))



##########################
#  Feature computation
##########################


class FeatureComputer:
    def __init__(self, documents, vocab=None, idf=None):
        """
        Initializes the FeatureComputer class.
        :param documents: Training or test dataset
        :param vocab: Vocabulary shared across datasets (for consistent feature representation)
        :param idf: Precomputed IDF values (for consistent TF-IDF computation across datasets)
        """
        self.docs = self.load_documents(documents)
        self.tokenized_docs = [set(word_tokenize(doc)) for doc, _ in self.docs]
        self.vocab = vocab if vocab is not None else self.extract_vocabulary()
        self.idf = idf if idf is not None else self.compute_idf()

    def extract_vocabulary(self):
        """
        Extracts a set of all unique words in the dataset.
        :return: Set of vocabulary words
        """
        vocab = set()
        for tokens in self.tokenized_docs:
            vocab.update(tokens)
        return vocab

    def compute_idf(self):
        """
        Computes the Inverse Document Frequency (IDF) for each word in the vocabulary.
        Formula: IDF(word) = log((N + 1) / (df + 1)) + 1
        where N is the total number of documents and df is the number of documents containing the word.
        :return: Dictionary with words as keys and their IDF values as values
        """
        num_docs = len(self.docs)
        term_document_map = {word: 0 for word in self.vocab}
        for tokens in self.tokenized_docs:
            for word in tokens:
                term_document_map[word] += 1
        idf = {word: log(num_docs / (1 + count)) + 1 for word, count in term_document_map.items()}
        return idf

    def compute_tfidf_features(self, document):
        """
        Computes the TF-IDF features for a given document.
        Formula:
        - TF(word) = Count(word) / TotalWords(document)
        - TF-IDF(word) = TF(word) * IDF(word)
        :param document: Text document to compute features for
        :return: List of TF-IDF values for all vocabulary words
        """
        words = word_tokenize(document)
        total_words = len(words)
        tf = {word: words.count(word) / total_words for word in set(words)}
        tfidf_features = [tf.get(word, 0) * self.idf.get(word, 0) for word in self.vocab]
        return tfidf_features

    def simple_features(self, document):
        """
        Computes basic features: number of sentences, average words per sentence, and average characters per word.
        :param document: Text document to compute features for
        :return: List of basic feature values
        """
        sentences = sent_tokenize(document)
        num_sent = len(sentences)
        mean_words = np.mean([len(word_tokenize(sent)) for sent in sentences])
        mean_chars = np.mean([len(word) for word in word_tokenize(document)])
        return [num_sent, mean_words, mean_chars]

    def best_features(self, document):
        """
        Computes additional features such as unique word count, stopword count, punctuation count, etc.
        :param document: Text document to compute features for
        :return: List of additional feature values
        """
        words = word_tokenize(document)
        total_words = len(words)
        unique_words = len(set(words))
        stopword_count = sum(1 for word in words if word.lower() in stop_words)
        punctuation_count = sum(1 for char in document if char in string.punctuation)
        capitalized_words = sum(1 for word in words if word.isupper())

        word_diversity = unique_words / total_words if total_words > 0 else 0
        avg_word_length = np.mean([len(word) for word in words]) if total_words > 0 else 0
        capitalization_ratio = capitalized_words / total_words if total_words > 0 else 0

        return [unique_words, stopword_count, word_diversity, punctuation_count, avg_word_length, capitalization_ratio]

    def compute_features(self, include_additional=False):
        """
        Computes features for all documents in the dataset.
        :param include_additional: Flag to include additional features and TF-IDF features
        :return: Feature matrix and corresponding labels
        """
        feature_set = []
        labels = []
        for doc, label in self.docs:
            features = self.simple_features(doc)
            if include_additional:
                features += self.best_features(doc)
                features += self.compute_tfidf_features(doc)
            feature_set.append(features)
            labels.append(label)
        return np.array(feature_set), np.array(labels)

    def load_documents(self, documents):
        """
        Loads documents into a format suitable for feature computation.
        :param documents: Dataset with columns "data" and "label"
        :return: List of tuples containing document text and label
        """
        return [(row["data"], row["label"]) for _, row in documents.iterrows()]

def read_data(file_path):
    """
    Reads tab-separated data files and returns them as pandas DataFrame.
    :param file_path: Path to the data file
    :return: DataFrame with columns "data" and "label"
    """
    return pd.read_csv(file_path, sep='\t', names=["data", "label"], skiprows=1)

##########################
#       Classifier
##########################

def main():
    """
    Main function to load data, compute features, train models, and output results.
    """
    train_data = read_data("train.tsv")
    test_data = read_data("test.tsv")

    feature_comp = FeatureComputer(train_data)
    test_feature_comp = FeatureComputer(test_data, vocab=feature_comp.vocab, idf=feature_comp.idf)

    # Simple feature computation
    train_X, train_y = feature_comp.compute_features()
    test_X, test_y = test_feature_comp.compute_features()

    # Extended feature computation
    best_train_X, _ = feature_comp.compute_features(include_additional=True)
    best_test_X, _ = test_feature_comp.compute_features(include_additional=True)

    # Simple model
    model_simple = LogisticRegression(max_iter=200)
    model_simple.fit(train_X, train_y)
    simple_accuracy = model_simple.score(test_X, test_y)
    simple_predictions = model_simple.predict(test_X)

    print(f"Accuracy with simple features: {simple_accuracy:.2f}")

    # Best model
    best_model = LogisticRegression(max_iter=200)
    best_model.fit(best_train_X, train_y)
    full_accuracy = best_model.score(best_test_X, test_y)
    full_predictions = best_model.predict(best_test_X)
    print(f"Accuracy with best features (including TF-IDF): {full_accuracy:.2f}")

    # Saving predictions with actual labels
    pd.DataFrame({"Prediction": simple_predictions, "Actual": test_y}).to_csv("simple_model_predictions.csv", index=False)
    pd.DataFrame({"Prediction": full_predictions, "Actual": test_y}).to_csv("best_model_predictions.csv", index=False)


if __name__ == "__main__":
    main()
