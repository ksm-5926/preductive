import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = {
    'packet_size': [500, 600, 1500, 2000, 100, 400, 1800, 1200, 300, 2500],
    'connection_duration': [30, 45, 10, 5, 100, 40, 8, 15, 70, 3],
    'intrusion': [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['packet_size', 'connection_duration']]
y = df['intrusion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_traffic_samples = pd.DataFrame({
    'packet_size': [1600, 350, 2200],
    'connection_duration': [7, 50, 3]
})

predictions = model.predict(new_traffic_samples)

for i, pred in enumerate(predictions):
    status = "Intrusion" if pred == 1 else "Normal"
    print(
        f"Traffic with packet_size={new_traffic_samples.iloc[i]['packet_size']} "
        f"and duration={new_traffic_samples.iloc[i]['connection_duration']} "
        f"is predicted as: {status}"
    )
