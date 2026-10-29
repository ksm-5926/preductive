import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'fever': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    'cough': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    'fatigue': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    'headache': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    'loss_of_appetite': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    'sick': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('sick', axis=1)
y = df['sick']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importances = model.feature_importances_

sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importance in Predicting Sickness')
plt.xlabel('Importance Score')
plt.ylabel('Behavioral Symptom')
plt.tight_layout()
plt.show()
