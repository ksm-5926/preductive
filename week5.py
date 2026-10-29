import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'heart_rate': [72, 85, 95, 110, 65, 98, 100, 105, 76, 88],
    'respiratory_rate': [16, 18, 22, 30, 14, 25, 28, 32, 17, 20],
    'skin_rash': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    'known_allergen_exposure': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    'swelling': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    'reaction_history': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    'allergic_reaction': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop('allergic_reaction', axis=1)
y = df['allergic_reaction']

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
plt.title("Symptom Importance for Allergic Reaction Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
