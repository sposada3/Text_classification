#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:00:19 2023

"""

# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Inspired by https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/
# Read the data
df = pd.read_csv('/Users/satposada/news.csv')

# Display the shape of the DataFrame
print("Shape:", df.shape)

# Display the first few rows of the DataFrame
print("Head:")
print(df.head())


# Separate features 
X = df['text']
labels = df['label']

# Split the data 
x_train,x_test,y_train,y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Preprocess features using TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform train set, transform test set
tfidf_train = vectorizer.fit_transform(x_train) 
tfidf_test = vectorizer.transform(x_test)

# Initialize classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier()),
    ('Linear SVM', LinearSVC()),
    ('Naive Bayes', MultinomialNB())
]

# Train and evaluate classifiers
for name, classifier in classifiers:
    
    # Create a pipeline
    pipeline = Pipeline([
        ('classifier', classifier)
    ])
   
    # Train the classifiers
    pipeline.fit(tfidf_train, y_train)

    # Make predictions on the testing set
    y_pred = pipeline.predict(tfidf_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {round(accuracy*100,2)}%')

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

   

   