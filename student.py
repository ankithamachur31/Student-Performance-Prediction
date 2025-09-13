import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Pre Processing
df=pd.read_csv("student_data.csv")

#Encoding Categorical Values
le=LabelEncoder()
df['Internet']=le.fit_transform(df["Internet"])
df['Passed']=le.fit_transform(df["Passed"])

#Feature Scaling
features=['StudyHours','Attendance','PastScore','SleepHours']
scalar=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scalar.fit_transform(df[features])

X=df_scaled[features] #Features
y=df_scaled['Passed'] #Target value

#Split the data
X_Train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train a model
model=LogisticRegression()
model.fit(X_Train,y_train)

#Make Prediction
y_pred=model.predict(X_test)

#Evaluating the model
print('Classification Report')
print(classification_report(y_test,y_pred))

#Confusion Matrix
conf_matrix=confusion_matrix(y_test,y_pred)

#Visualize Result
plt.figure(figsize=(6,4))

sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


print("-------Predict Your Score-------")
try:
    study_hours=float(input("Enter Study Hours: "))
    attendance=float(input("Enter Attendance: "))
    past_score=float(input("Enter Past Score: "))
    sleep_hours=float(input("Enter Sleep Hours: "))

    user_input_df=pd.DataFrame([{
        'StudyHours':study_hours,
        'Attendance':attendance,
        'PastScore':past_score,
        'SleepHours':sleep_hours
        
    }])

    user_input_scaled=scalar.transform(user_input_df)
    prediction=model.predict(user_input_scaled)[0]
    result="Pass" if prediction==1 else "Fail"
    print(f"Prediction Based on input: {result}")
except Exception as e:
    print("An error occured ",e)