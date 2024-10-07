# Amazon Customer Review Sentiment Analysis

## Objective
The objective of this project is to classify Amazon customer reviews into positive and negative sentiments by evaluating the performance of a variety of machine learning algorithms. This project is essential for helping companies make better decisions based on sentiment analysis results, improving their services, and raising customer satisfaction levels.

## Problem Statement
Customer feedback is a key factor for any business to grow and maintain customer satisfaction. However, analyzing thousands of customer reviews manually is challenging and time-consuming. This project aims to automate the process of sentiment analysis by classifying customer reviews as positive or negative, using machine learning and deep learning algorithms. This allows companies to quickly understand customer sentiments and respond promptly.

## Dataset
The dataset used in this project is the **Amazon Fine Food Reviews Dataset**, which contains customer reviews for Amazon products. The dataset comprises 10,000 training samples and 200 test samples that were preprocessed for our analysis. 
[Dataset Source](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## Working of the Project

### Preprocessing:
1. **Porter Stemming**: Used as part of the vectorization pipeline to condense vocabulary size and improve the model's ability to identify relevant patterns.
2. **TF-IDF Vectorization**: Applied to convert text into numerical features.
3. **Stop Word Removal**: Used to filter out irrelevant common words and focus on key features.

### Model Training and Evaluation:
Several machine learning algorithms are tested for sentiment analysis:

1. **K-Nearest Neighbors (KNN)**: Works by classifying data points based on the most common label among its nearest neighbors. However, it performed poorly due to high-dimensional feature space.
   
2. **Logistic Regression**: A linear model used for binary classification. It achieved a high test accuracy (95.5%) after tuning hyperparameters such as penalty and IDF weighting.
   
3. **Support Vector Machine (SVM)**: This algorithm finds the optimal hyperplane for classification, achieving the best accuracy of 96%.
   
4. **Multinomial Naive Bayes**: A probabilistic classifier that worked well for text classification, achieving 91% test accuracy.
   
5. **Bidirectional LSTM (BILSTM)**: A deep learning model used for capturing long-term dependencies in text. Though it showed moderate accuracy (88.7%), the shallow architecture contributed to faster training times.

### Hyperparameter Tuning:
Each model underwent extensive hyperparameter tuning and cross-validation to achieve the best performance. Some key parameters included the n-gram range, stop word removal, and the use of IDF weighting in TF-IDF vectorization.

## Python Notebooks and Files:
This project contains the following notebooks:
  
1. **`group8.ipynb`**:
   - Data preprocessing steps such as stemming, tokenization, stop word removal, and vectorization using TF-IDF.
   - Exploratory Data Analysis (EDA) including visualizations and frequency analysis.
   - Data distribution and insights to inform further stages of the project.
  
2. **`mlalgo.ipynb`**:
   - Implementation of machine learning models such as KNN, Logistic Regression, SVM, and Naive Bayes.
   - Hyperparameter tuning and cross-validation for optimal performance of each model.

3. **`bilstm.ipynb`**:
   - Implements a Bidirectional LSTM (BILSTM) model for text classification.
   - Trains and evaluates the BILSTM model using preprocessed customer reviews.
   - Performance metrics and confusion matrix are provided to analyze the results.

## Conclusion
The Support Vector Machine (SVM) algorithm achieved the best accuracy at 96%, closely followed by Logistic Regression. The Bidirectional LSTM model performed decently with an accuracy of 88.7% and a shorter training time, which makes it suitable for cases where training speed is a concern. KNN underperformed due to the high-dimensional nature of the feature space.

This project highlights the importance of customer sentiment analysis and how machine learning can help companies respond promptly to feedback, improving overall customer satisfaction. Future improvements could include exploring more advanced deep learning architectures, implementing ensemble methods, and analyzing temporal trends in customer reviews for better decision-making.
