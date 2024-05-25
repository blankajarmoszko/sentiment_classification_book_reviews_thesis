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

# Vectorize the text using Bag-of-Words
vectorizer_bow = CountVectorizer()
X_train_vectorized_bow = vectorizer_bow.fit_transform(train_data)
X_test_vectorized_bow = vectorizer_bow.transform(test_data)

# Classifier - Algorithm - SVM
# Fit the training dataset on the classifier
SVM_bow = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', verbose=True)
SVM_bow.fit(X_train_vectorized_bow, train_labels)

# Predictions
train_predictions = SVM_bow.predict(X_train_vectorized_bow)
test_predictions = SVM_bow.predict(X_test_vectorized_bow)

# Print accuracy on train and test
train_accuracy = accuracy_score(train_predictions, train_labels)
test_accuracy = accuracy_score(test_predictions, test_labels)
print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Print classification report for train and test
print("\nClassification Report for Train Set:")
print(classification_report(train_labels, train_predictions))

print("\nClassification Report for Test Set:")
print(classification_report(test_labels, test_predictions))

# Print confusion matrices for train and test
print("\nConfusion Matrix for Train Set:")
print(confusion_matrix(train_labels, train_predictions))

print("\nConfusion Matrix for Test Set:")
print(confusion_matrix(test_labels, test_predictions))
