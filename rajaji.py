import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(0)

n_samples = 200
pressure = np.random.uniform(990, 1020, n_samples)  # Pressure in hPa
humidity = np.random.uniform(40, 100, n_samples)  # Humidity in percentage
wind_speed = np.random.uniform(0, 20, n_samples)  # Wind speed in m/s
temperature = 15 + 0.2 * pressure - 0.05 * humidity + 0.1 * wind_speed + np.random.normal(0, 2, n_samples)  # Temperature in Celsius

weather_data = pd.DataFrame({
    'Pressure': pressure,
    'Humidity': humidity,
    'WindSpeed': wind_speed,
    'Temperature': temperature
})
X = weather_data[['Pressure', 'Humidity', 'WindSpeed']]
y = weather_data['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.show()

new_data = np.array([[1015, 75, 5]])  # Example: pressure=1015 hPa, humidity=75%, wind speed=5 m/s
predicted_temperature = model.predict(new_data)
print(f'Predicted Temperature: {predicted_temperature[0]:.2f}°C')
