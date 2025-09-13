import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


df=pd.read_excel("D:\Student_Mental_Analysis-Project-3\Student_Mental_Health_Datasetss.xlsx")
features=['Age','Gender','Education_level','Headache','Sleep_Hours_Per_Night','Screen Time','Stress_Level','Depression_Score',
          'Anxiety_Score','Financial_Stress','Physical_Activity','Diet_Quality',
          'Relationship_Status','Substance_Use','Chronic_Illness','Most_Used_Platform']
df['mental_score']=df['Stress_Level']+df['Depression_Score']+df['Anxiety_Score']
df['healthy_lifestyle']=df['Sleep_Hours_Per_Night']-df['Headache']+df['Physical_Activity'].map({'Low':0,'Moderate':1,'High':2})
features=features+['mental_score','healthy_lifestyle']
target='Mental_Health_Score'

def get_score(Score):
    if Score<=5:
        return 0
    elif 5<=Score<=8:
        return 1
    else:
        return 2

df['risk_level'] = df[target].apply(get_score)

print(df['risk_level'].value_counts())

df=df.dropna(subset=features+[target])


x=df[features]
y=df['risk_level']

categorical_features = ['Gender', 'Education_level', 'Physical_Activity', 'Diet_Quality',
                        'Relationship_Status', 'Substance_Use', 'Chronic_Illness', 'Most_Used_Platform']

numerical_features = list(set(features) - set(categorical_features))


preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Model pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('smote',SMOTE(random_state=42)),
     ('rf',RandomForestClassifier())
])
xgb_pipeline = Pipeline([
    ('preprocess', preprocessor),
     ('xgb',XGBClassifier( eval_metric='mlogloss', random_state=42))
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

params={
    'rf__n_estimators':[100,200],
    'rf__max_depth':[10,20],
    'rf__min_samples_split':[2,4],
    'rf__min_samples_leaf':[1,2]
}
grid=GridSearchCV(pipeline,params,cv=5,scoring='f1_macro')
grid.fit(x_train,y_train)
y_pred=grid.predict(x_test)

#Tuning
params_xgb = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__subsample': [0.8, 1.0]
}

grid_xgb = GridSearchCV(xgb_pipeline, params_xgb, cv=5, scoring='f1_macro', n_jobs=-1)
grid_xgb.fit(x_train, y_train)
# Predict and evaluate
xgb_pred = grid_xgb.predict(x_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_conf = confusion_matrix(y_test, xgb_pred)
disp=ConfusionMatrixDisplay(xgb_conf,display_labels=['Low Risk','Medium Risk','High Risk'])
disp.plot(cmap='Blues')
plt.show()
xgb_classify_report = classification_report(y_test, xgb_pred)

print("\n")

print(f"XGBoost Confusion Matrix:{xgb_conf}")
print(f"XGBoost Classification Report:{xgb_classify_report}")
print("Best Parameters:", grid_xgb.best_params_)
print("Best Accuracy:", grid_xgb.best_score_)

vot_clf = VotingClassifier(estimators=[
    ('rf', grid.best_estimator_),
    ('xgb', grid_xgb.best_estimator_)
], voting='soft')

vot_clf.fit(x_train, y_train)
y_pred=vot_clf.predict(x_test)
vot_accuracy=accuracy_score(y_test,y_pred)
vot_conf=confusion_matrix(y_test,y_pred)
vot_classify_report=classification_report(y_test,y_pred)

print(f"Accuracy:{vot_accuracy}")
print(f"Confusion Matrix:{vot_conf}")
print(f"Classification Report:{vot_classify_report}")

print("\n")

print("Prediction of Student Mental Health")
user_input={}
print("\nPlease enter the details of the student")
student_name=input("Name:")
user_input['Age'] = int(input("Age: "))
user_input['Gender'] = input("Gender (Male/Female): ")
user_input['Education_level'] = input("Education Level (Class 9-12/BA/Mtech/etc..): ")
user_input['Headache'] = int(input("Headache(0-No,1-Mild,2-Frequent: "))
user_input['Sleep_Hours_Per_Night'] = float(input("Average Sleep Hours per Night: "))
user_input['Screen Time'] = float(input("Screen Time: "))
user_input['Stress_Level'] = int(input("Stress Level (0-10): "))
user_input['Depression_Score'] = int(input("Depression Score (0-10): "))
user_input['Anxiety_Score'] = int(input("Anxiety Score (0-10): "))
user_input['Financial_Stress'] = int(input("Financial Stress Level(0-10): "))
user_input['Physical_Activity'] = input("Physical Activity (Low/Moderate/High): ")
user_input['Diet_Quality'] = input("Diet Quality (Poor/Average/Good): ")
user_input['Relationship_Status'] = input("Relationship Status (Single/Married/In a relationship): ")
user_input['Substance_Use'] = input("Substance Use (Never/Occasionally/Frequently): ")
user_input['Chronic_Illness'] = input("Chronic Illness (Yes/No): ")
user_input['Most_Used_Platform'] = input("Most Used Platform (Instagram/Facebook/etc.): ")


user_df=pd.DataFrame([user_input])

user_df['mental_score'] = user_df['Stress_Level'] + user_df['Depression_Score'] + user_df['Anxiety_Score']
user_df['healthy_lifestyle'] = (
    user_df['Sleep_Hours_Per_Night'] - user_df['Headache'] +
    user_df['Physical_Activity'].map({'Low': 0, 'Moderate': 1, 'High': 2}).fillna(0)
)

user_df=user_df[features]
print("\nDetails of the Student")
print("\nStudent Name:",student_name)
for key,value in user_input.items():
    print(f"{key}:{value}")
recommendations={
    'Low Risk': "Keep up the good work! Continue maintaining a balanced lifestyle.",
    'Medium Risk': "Take proactive steps now—small changes can prevent bigger issues later.",
    'High Risk': "Seek immediate support. Consider consulting a counselor or mental health professional."
}

risk_labels={0:'Low Risk',1:'Medium Risk',2:'High Risk'}

predictions=vot_clf.predict(user_df)
predicted_labels = risk_labels[predictions[0]]
print(f"\nPrediction of Student Mental Health:",predicted_labels)
print("Recommendation:",recommendations[predicted_labels])


