import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = {
    'day': range(1, 31),
    'temperature': [
        22, 21, 23, 24, 25, 23, 22, 21, 22, 24,
        26, 25, 27, 28, 29, 27, 26, 25, 24, 23,
        22, 23, 24, 25, 26, 27, 26, 25, 24, 23
    ]
}

df = pd.DataFrame(data)

df['temp_prev_day'] = df['temperature'].shift(1)
df.dropna(inplace=True)

X = df[['temp_prev_day']]
y = df['temperature']

train_size = len(df) - 5
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

plt.plot(df['day'][train_size:], y_test, label='Actual Temp')
plt.plot(df['day'][train_size:], y_pred, label='Predicted Temp', linestyle='--')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Weather Forecasting - Temperature Prediction')
plt.legend()
plt.show()
