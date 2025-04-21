import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("Tesla Stock Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Use only the 'Open' price
data = df[['Open']].values

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences of 20 days to predict the next day's open price
sequence_length = 20
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i].flatten())
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y).ravel()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate
y_pred = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.6f}")

# Predict the opening price for April 26
last_20_days = scaled_data[-20:].flatten().reshape(1, -1)
predicted_scaled = mlp.predict(last_20_days)
predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

print(f"Predicted Opening Price on April 26: ${predicted_price[0][0]:.2f}")
