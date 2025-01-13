import pandas as pd

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Display first few rows
print(data.head())

# Summary statistics
print(data.info())
print(data.describe())i

# Check for missing values
print(data.isnull().sum())

# Example: Fill missing ages with the median
data['Age'] = data['Age'].fillna(data['Age'].median())

# Drop unnecessary columns (e.g., 'Cabin' if too many missing values)
data = data.drop(['Cabin', 'Name', 'Ticket'], axis=1)


# Example: Convert 'Sex' to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)



# Model Building
X = data.drop("Survived", axis=1)
y = data["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed classification report
print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Survival countplot
sns.countplot(data['Survived'])
plt.show()

import joblib

# Save the model
joblib.dump(model, "titanic_model.pkl")

# Load the model
loaded_model = joblib.load("titanic_model.pkl")


def predict_survival(passenger_data):
    prediction = loaded_model.predict([passenger_data])
    return "Survived" if prediction[0] == 1 else "Did not survive"

# Example usage
try:
    # Predict using sample passenger data
    sample_passenger = [1, 4, 0, 0, 4, 0, 0, 1,0]  # Includes Embarked_Q and Embarked_S

    print(predict_survival(sample_passenger))
except Exception as e:
    print("Error:", e)



