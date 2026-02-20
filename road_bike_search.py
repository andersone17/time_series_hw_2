# Quick script to assess road bike search term using ARIMA / SARAIM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import mean_squared_error
import numpy as np

# Load Data
df = pd.read_csv('data/road_bike_search_term.csv')

plot_acf(df['road bike'], lags=15)

plot_pacf(df['road bike'], lags=15)

# Differeneced PACF and ACF
plot_acf(df['road bike'].diff().dropna(), lags=15)
plot_pacf(df['road bike'].diff().dropna(), lags=15)

# Auto ARIMA
from pmdarima import auto_arima
model = auto_arima(df['road bike'], seasonal=True, m=12)
print(model.summary())

train = df[:-24]
test = df[-24:]

# ARIMA MODEL
P = 1
D = 1
Q = 1
P_S = 1
D_S = 0
Q_S = 1
S = 12

model = ARIMA(train['road bike'], order=(P,D,Q), seasonal_order=(P_S, D_S, Q_S, S))
model_fit = model.fit()
print(model_fit.summary())

# Predictions
forecast = model_fit.get_forecast(steps=24)
predictions = forecast.predicted_mean
ci_lower, ci_upper = forecast.conf_int().T.values

# Add to Test Set
test['predictions'] = predictions
test['ci_lower'] = ci_lower
test['ci_upper'] = ci_upper

# Concat Train and Test
full_data = pd.concat([train, test])

# Plot with CIs
plt.figure(figsize=(12, 6))
plt.plot(full_data['Time'], full_data['road bike'], label='Actual', marker='o')
plt.plot(test['Time'], test['predictions'], label='Predicted', marker='o')
plt.fill_between(test['Time'], test['ci_lower'], test['ci_upper'], color='gray', alpha=0.3, label='95% CI')
plt.title('SARIMA: Road Bike Search Term Forecast')
plt.xlabel('Time')
plt.ylabel('Search Volume')
plt.legend()
plt.show()



# Auto ARIMA
from pmdarima import auto_arima
model = auto_arima(df['road bike'], seasonal=False)
print(model.summary())

P = 2
D = 1
Q = 2
model = ARIMA(train['road bike'], order=(P,D,Q))
model_fit = model.fit()
print(model_fit.summary())

# Predictions
forecast = model_fit.get_forecast(steps=24)
predictions = forecast.predicted_mean
ci_lower, ci_upper = forecast.conf_int().T.values

# Add to Test Set
test['predictions_arima'] = predictions
test['ci_lower_arima'] = ci_lower
test['ci_upper_arima'] = ci_upper
full_data = pd.concat([train, test])
# Plot with CIs
plt.figure(figsize=(12, 6))
plt.plot(full_data['Time'], full_data['road bike'], label='Actual', marker='o')
plt.plot(test['Time'], test['predictions_arima'], label='Predicted', marker='o')
plt.fill_between(test['Time'], test['ci_lower_arima'], test['ci_upper_arima'], color='gray', alpha=0.3, label='95% CI')
plt.title('ARIMA: Road Bike Search Term Forecast')
plt.xlabel('Time')
plt.ylabel('Search Volume')
plt.legend()
plt.show()