# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Read the data file into a Pandas DataFrame
df = pd.read_csv('/Users/blankajarmoszko/PycharmProjects/thesis/data/df_cleaned.csv')
def get_wrong_reviews(model, x_test, test_labels, name):
    # Make predictions on the test data
    y_pred = model.predict(X_test_tfidf)

    # Identify misclassifications
    misclassified_indices = [i for i in range(len(test_labels)) if test_labels.iloc[i] != y_pred[i]]

    # Retrieve misclassified entries
    misclassified_entries = df.iloc[misclassified_indices].copy()

    # Add predicted labels to the DataFrame
    misclassified_entries['Predicted Label'] = y_pred[misclassified_indices]

    # Save DataFrame with only misclassified entries as csv
    file_path = f"/Users/blankajarmoszko/PycharmProjects/thesis/models/missclassified_data/{name}.csv"
    misclassified_entries.to_csv(file_path, index=False)
    print("Done")
# Function to map stars to sentiment
def map_sentiment(stars_received):
    if stars_received <= 3:
        return 0
    elif stars_received <= 4:
        return 1
    else:
        return 2


# Mapping stars to sentiment into three categories
df['sentiment'] = [map_sentiment(x) for x in df['star_rating']]

# Drop rows with NaN values in the 'cleaned_text' column
df = df.dropna(subset=['cleaned_text'])

# Reset index after removing rows
df.reset_index(drop=True, inplace=True)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, ngram_range=(1,3))
X_train_tfidf = vectorizer.fit_transform(train_data)
X_test_tfidf = vectorizer.transform(test_data)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define a shallow decision tree as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=2)

# Reduce the number of estimators to make training faster
classifier = AdaBoostClassifier( base_estimator, n_estimators=50)

# Fit the classifier
classifier.fit(X_train_tfidf, train_labels)

# Predictions on training set
train_preds = classifier.predict(X_train_tfidf)

# Predictions on test set
test_preds = classifier.predict(X_test_tfidf)

# Classification report and confusion matrix for training set
print("Training Set:")
print("Classification Report:")
print(classification_report(train_labels, train_preds))
print("Confusion Matrix:")
print(confusion_matrix(train_labels, train_preds))

# Classification report and confusion matrix for test set
print("\nTest Set:")
print("Classification Report:")
print(classification_report(test_labels, test_preds))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds))

get_wrong_reviews(classifier,X_test_tfidf,test_labels, "ada_tfidf")