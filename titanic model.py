import pandas as pd


data = pd.read_csv("Titanic-Dataset.csv")


print(data.head())


print(data.info())
print(data.describe())i


print(data.isnull().sum())

data['Age'] = data['Age'].fillna(data['Age'].median())

data = data.drop(['Cabin', 'Name', 'Ticket'], axis=1)



data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})


data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

X = data.drop("Survived", axis=1)
y = data["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report


print("Accuracy:", accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt


sns.countplot(data['Survived'])
plt.show()

import joblib


joblib.dump(model, "titanic_model.pkl")


loaded_model = joblib.load("titanic_model.pkl")


def predict_survival(passenger_data):
    prediction = loaded_model.predict([passenger_data])
    return "Survived" if prediction[0] == 1 else "Did not survive"


try:
    sample_passenger = [1, 4, 0, 0, 4, 0, 0, 1,0]  
    print(predict_survival(sample_passenger))
except Exception as e:
    print("Error:", e)



