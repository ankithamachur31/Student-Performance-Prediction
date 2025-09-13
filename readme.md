# **Student Performance Prediction using Logistic Regression**

## **Project Overview**

This project aims to predict whether a student will pass or fail based on several key factors such as study hours, class attendance, past scores, and sleep hours. A Logistic Regression model is trained on a dataset of student information to make these predictions. The project also includes a feature to predict the outcome for a new student based on user-provided input.

## **How It Works**

The script performs the following steps:

1. **Loads** the student data from a CSV file.  
2. **Preprocesses** the data by encoding categorical variables and scaling numerical features.  
3. **Splits** the data into training and testing sets.  
4. **Trains** a Logistic Regression model on the training data.  
5. **Evaluates** the model's performance using a classification report and a confusion matrix.  
6. **Visualizes** the confusion matrix for an intuitive understanding of the results.  
7. **Prompts** the user to enter new student data and predicts the pass/fail outcome.

## **Dataset**

The project requires a CSV file named student\_data.csv with the following columns:

| Column | Type | Description |
| :---- | :---- | :---- |
| **StudyHours** | Numeric | The number of hours the student studies per day. |
| **Attendance** | Numeric | The student's attendance percentage. |
| **PastScore** | Numeric | The student's score in a previous exam. |
| **SleepHours** | Numeric | The number of hours the student sleeps per day. |
| **Internet** | Categorical | Whether the student has internet access ('Yes'/'No'). |
| **Passed** | Categorical | The target variable; whether the student passed ('Yes'/'No'). |

Here is an example of what student\_data.csv might look like:

StudyHours,Attendance,PastScore,SleepHours,Internet,Passed  
4.5,85,75,7,Yes,Yes  
2.1,60,45,6,No,No  
6.8,95,88,8,Yes,Yes  
3.2,70,60,7,Yes,No

## **Getting Started**

Follow these instructions to get the project running on your local machine.

### **Prerequisites**

* Python 3.x  
* pip (Python package installer)

### **Installation**

1. Clone this repository or download the source code.  
2. Install the required Python libraries by running the following command in your terminal:  
   pip install pandas numpy scikit-learn matplotlib seaborn

## **Running the Project**

1. Make sure you have the student\_data.csv file in the same directory as the Python script.  
2. Open your terminal or command prompt.  
3. Navigate to the directory containing the project files.  
4. Run the script using the following command:  
   python your\_script\_name.py

5. The script will first print the model's evaluation metrics and display the confusion matrix plot.  
6. After you close the plot, the terminal will prompt you to enter new data to make a prediction.

## **Code Explanation**

### **1\. Importing Libraries**

The script starts by importing the necessary libraries for data manipulation, machine learning, and visualization.

import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.model\_selection import train\_test\_split  
from sklearn.linear\_model import LogisticRegression  
from sklearn.metrics import classification\_report, confusion\_matrix  
import matplotlib.pyplot as plt  
import seaborn as sns

### **2\. Data Loading and Preprocessing**

* **Load Data**: The student\_data.csv file is loaded into a pandas DataFrame.  
* **Label Encoding**: The Internet and Passed columns contain categorical text ('Yes'/'No'). Machine learning models require numerical input, so LabelEncoder is used to convert these strings into numbers (0 and 1).  
* **Feature Scaling**: The numerical features (StudyHours, Attendance, PastScore, SleepHours) are scaled using StandardScaler. This standardizes the features to have a mean of 0 and a standard deviation of 1, which helps the logistic regression model converge faster and perform better.

### **3\. Data Splitting**

The dataset is split into features (X) and the target variable (y). It is then divided into a training set (80% of the data) and a testing set (20%) to evaluate the model's performance on unseen data.

X \= df\_scaled\[features\]  
y \= df\_scaled\['Passed'\]  
X\_Train, X\_test, y\_train, y\_test \= train\_test\_split(X, y, test\_size=0.2, random\_state=42)

### **4\. Model Training**

A LogisticRegression model is initialized and trained using the fit() method on the training data (X\_Train, y\_train).

model \= LogisticRegression()  
model.fit(X\_Train, y\_train)

### **5\. Model Evaluation**

* **Classification Report**: This report shows the main classification metrics: precision, recall, and F1-score for each class (Pass/Fail).  
* **Confusion Matrix**: A confusion matrix is generated to provide a more detailed breakdown of prediction accuracy, showing correct and incorrect predictions. The matrix is then visualized as a heatmap using seaborn and matplotlib for better interpretation.

### **6\. Predicting for a New Student**

The script includes a user-interactive section that:

* Prompts the user to input values for the features.  
* Places the input into a pandas DataFrame.  
* **Scales the user input using the same StandardScaler instance (scalar) that was fitted on the training data.** This is a crucial step to ensure the prediction is accurate.  
* Uses the trained model to predict the outcome.  
* Prints the final result ("Pass" or "Fail") to the console.

## **Conclusion**

This project successfully demonstrates a complete machine learning workflow: from data preprocessing and model training to evaluation and deployment for making live predictions. The logistic regression model provides a simple yet effective baseline for this classification task.