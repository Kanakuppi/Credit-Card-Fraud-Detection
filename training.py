import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.svm import SVC

dataset=pd.read_csv('creditcard.csv')

fraud=dataset[dataset.Class==1]
legit=dataset[dataset.Class==0]

legit_sample=legit.sample(n=492)

balanced_df=pd.concat([legit_sample,fraud],axis=0)

X=balanced_df.drop(columns='Class',axis=1)
Y=balanced_df['Class']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=42)

Logistic_model=LogisticRegression()
Logistic_model.fit(X_train,Y_train)

predict=Logistic_model.predict(X_test)

accuracy=accuracy_score(predict,Y_test)

decision=RandomForestClassifier()
decision.fit(X_train,Y_train)

accuracy_decision=accuracy_score(decision.predict(X_test),Y_test)

import joblib
import os

LOGISTIC_MODEL_FILE = "logistic_model.pkl"
DECISION_MODEL_FILE = "decision_model.pkl"
# Assuming you have a pipeline for preprocessing X
# If you don't have a specific pipeline object, you might need to create one
# based on the transformations applied to X before training the models.
# For now, I'll assume you have a pipeline object called 'pipeline'
# If not, you'll need to replace 'pipeline' with the actual preprocessing steps or create a pipeline.
# Example: If you used StandardScaler on X, you could save the scaler.
# For this example, I'll assume a pipeline object exists.
# PIPELINE_FILE = "pipeline.pkl"

# Save the Logistic Regression model
if not os.path.exists(LOGISTIC_MODEL_FILE):
    joblib.dump(Logistic_model, LOGISTIC_MODEL_FILE)
    print(f"Logistic Regression model saved to {LOGISTIC_MODEL_FILE}")
else:
    print("Logistic Regression model already exists.")

# Save the Random Forest model
if not os.path.exists(DECISION_MODEL_FILE):
    joblib.dump(decision, DECISION_MODEL_FILE)
    print(f"Random Forest model saved to {DECISION_MODEL_FILE}")
else:
    print("Random Forest model already exists.")

# If you have a pipeline, uncomment and adapt the following lines
# if not os.path.exists(PIPELINE_FILE):
#     joblib.dump(pipeline, PIPELINE_FILE)
#     print(f"Pipeline saved to {PIPELINE_FILE}")
# else:
#     print("Pipeline already exists.")