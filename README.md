# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Define Problem

Identify the independent variables (X) and target labels (Y) in the dataset.

2.Load Dataset

Import and preprocess the data (e.g., handle missing values, encode categorical variables).

3.Split Data

Divide the dataset into training and testing subsets.

4.Initialize the Model

Use SGDClassifier from sklearn. Configure it to use the desired loss function such as:
loss='hinge' for SVM,
loss='log_loss' for logistic regression, or
loss='modified_huber' for robust classification.

5.Train the Model

Fit the SGDClassifier on the training dataset.

6.Evaluate the Model

Use metrics like accuracy, confusion matrix, precision, recall, and F1-score for evaluation.

7.Make Predictions

Predict the target class for unseen data.

8.Fine-Tune Parameters (Optional)

Adjust hyperparameters like learning rate, regularization term, or the number of iterations to improve performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: vishal.v
RegisterNumber: 24900179 
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = load_iris()
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Display the first few rows of the dataset
print(df.head())
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
![image](https://github.com/user-attachments/assets/02fc6cff-b6b1-4f89-9186-6bef12f2dd7b)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
