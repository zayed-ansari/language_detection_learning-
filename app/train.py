import pandas as pd 
import numpy as np 
import mlflow
import re
import pickle
import mlflow.sklearn


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.simplefilter('ignore')

mlflow.set_experiment("language-detection")
mlflow.sklearn.autolog()

le =LabelEncoder()
df = pd.read_csv('app/Language Detection.csv')
X = df['Text']
y = df['Language']
y = le.fit_transform(y)

data_list = [] 
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    data_list.append(text)

X_train, X_test, y_train, y_test= train_test_split(data_list,y, test_size=0.20, random_state=42)

# -------- Naive Model ------- #
def naive_baye(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="NaiveBaye"):
        pipe_nb = Pipeline([
            ('vectorizer', CountVectorizer()), 
            ('multinomialNB', MultinomialNB())])
        pipe_nb.fit(X_train, y_train)

        y_pred_nb= pipe_nb.predict(X_test)
        ac_nb = accuracy_score(y_test, y_pred_nb)

        # Log metric
        mlflow.log_metric("accuracy", ac_nb)

        # Log model 
        mlflow.sklearn.log_model(pipe_nb, 'model')

        mlflow.log_param("ngram_range", (1,1))
        mlflow.log_param("model", "MultinomialNB")
        print("Accuracy of Naive Bayes", ac_nb)
        with open('app/model/naive_trained_pipeline-0.1.0.pkl', 'wb') as f:
            pickle.dump(pipe_nb, f)

# ----- Logistic Regression Model ----- #
def logistic_regression(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="LogisticRegression"):
        pipe_lr = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("logistic_regression", LogisticRegression(max_iter=1000))
    ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)
        ac_lr = accuracy_score(y_test, y_pred_lr)

        # Log metric 
        mlflow.log_metric('accuracy', ac_lr)

        mlflow.sklearn.log_model(pipe_lr, 'model')
        # Log parameters
        mlflow.log_param("ngram_range", (1,1))
        mlflow.log_param("model", "LogisticRegression")
        print("Accuracy of Logistic Regression", ac_lr)
        with open('app/model/logistic_trained_pipeline-0.1.0.pkl', 'wb') as f:
            pickle.dump(pipe_lr, f)


# ------- Linear SVC Model ------- #
def linear_svm(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="LinearSVC"):
        pipe_svm = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('svc_model', LinearSVC())
        ])
        pipe_svm.fit(X_train, y_train)
        y_pred_svm = pipe_svm.predict(X_test)
        ac_svm = accuracy_score(y_test, y_pred_svm)
        # Log metric 
        mlflow.log_metric('accuracy', ac_svm)
        mlflow.sklearn.log_model(pipe_svm, 'model')

        mlflow.log_param("ngram_range", (1,1))
        mlflow.log_param("model", "LinearSVC")
        
        print("Accuracy of Linear SVC", ac_svm)
        with open('app/model/svm_trained_pipeline-0.1.0.pkl', 'wb') as f:
            pickle.dump(pipe_svm, f)
    

if __name__=="__main__":
    naive_baye(X_train, y_train, X_test, y_test)
    logistic_regression(X_train, y_train, X_test, y_test)
    linear_svm(X_train, y_train, X_test, y_test)