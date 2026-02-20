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

# Variables
time = df['Time']
series = df['road bike']

# Viz of Series
plt.style.use('ggplot')
plt.plot(time, series)
plt.show()

# ACF / PACF
acf = plot_acf(series, lags=50)
pacf = plot_pacf(series, lags=50)

# Differeneced PACF and ACF
acf = plot_acf(series.diff().dropna(), lags=50)
pacf = plot_pacf(series.diff().dropna(), lags=50)

# Auto SARIMA
from pmdarima import auto_arima
model = auto_arima(series, seasonal=True, m=12)
print(model.summary())
# Auto Select is (1,1,1)x(1,0,1)x12

# Auto ARIMA
from pmdarima import auto_arima
model = auto_arima(df['road bike'], seasonal=False)
print(model.summary())
# Auto Select is (3,1,5)

# Split Train and Test
dataset_len = len(df)
test_size = .2
test_size = round(dataset_len * test_size)
train = df[:-test_size]
test = df[-test_size:]


##################################################################################
# SARIMA MODEL -- Based on Auto ARIMA

def ARIMA_MODEL(train_set, test_set, series_col, date_col, p_d_q, P_D_Q_S=None):
    '''function to build arima and plot against actual data'''
    # ARIMA Function
    p, d, q = p_d_q[0], p_d_q[1], p_d_q[2] 
    if P_D_Q_S:
        p_s, d_s, q_s, s = P_D_Q_S[0], P_D_Q_S[1], P_D_Q_S[2], P_D_Q_S[3]
    # Build Model
    if P_D_Q_S:
        model = ARIMA(train_set[series_col], order=(p_d_q), seasonal_order=(P_D_Q_S))
    else: 
        model = ARIMA(train_set[series_col], order=(p_d_q))
    model_fit = model.fit()
    print(model_fit.summary())

    # Predictions
    forecast = model_fit.get_forecast(steps=test_set.shape[0])
    predictions = forecast.predicted_mean
    ci_lower, ci_upper = forecast.conf_int().T.values
    
    # Transform if Log Used
    if 'log' in series_col:
        predictions = np.exp(predictions)
        ci_lower = np.exp(ci_lower)
        ci_upper = np.exp(ci_upper)
        series_col_name = series_col.replace('log_', '')

    # Add to Test Set
    test_set['forecast'] = predictions
    test_set['ci_lower'] = ci_lower
    test_set['ci_upper'] = ci_upper

    # Concat Train and Test for Viz
    full_data = pd.concat([train_set, test_set], ignore_index=True)

    # Plot with CIs
    plt.figure(figsize=(12, 6))
    plt.plot(full_data[date_col], full_data[series_col_name], label='Actual', marker='o')
    plt.plot(test_set[date_col], test_set['forecast'], label='Forecasted', marker='o')
    plt.fill_between(test_set[date_col], test_set['ci_lower'], test['ci_upper'], color='gray', alpha=0.3, label='95% CI')
    plt.title('ARIMA Forecast')
    plt.xlabel(date_col)
    plt.ylabel(series_col_name)
    plt.legend()
    plt.show()


# 1,1,1 x 1,0,1,12
ARIMA_MODEL(
    train_set = train.copy(), 
    test_set = test.copy(), 
    series_col='road bike',
    date_col='Time',
    p_d_q=(1,1,1), 
    P_D_Q_S=(1,0,1,12)
)

# 2,1,2
ARIMA_MODEL(
    train_set = train.copy(), 
    test_set = test.copy(), 
    series_col='road bike',
    date_col='Time',    
    p_d_q=(2,1,2)
)

# Testing...
ARIMA_MODEL(
    train_set = train.copy(), 
    test_set = test.copy(), 
    series_col='road bike',
    date_col='Time',
    p_d_q=(5,1,5)
)


# Now lets do some testing with the canyon data...
# Load & Preprocess Data
sales = pd.read_csv('data/sales_data.csv')
sales['date'] = pd.to_datetime(sales['date'], errors='coerce')
sales['qty'] = pd.to_numeric(sales['qty'], errors='coerce')
sales['val'] = pd.to_numeric(sales['val'], errors='coerce')
# Subset and Group
sales = sales[~sales['world'].isin(['gr', 'na'])]
sales = sales.groupby('date').sum().reset_index().drop(columns=['world'])
sales.sort_values('date', inplace=True)
# Correct sales anaomaly
sales.rename(columns={'qty' : 'order_quantity', 'val' : 'order_value'}, inplace=True)
sales.loc[sales['order_quantity'].idxmax(), 'order_quantity'] = sales.loc[sales['order_quantity'].idxmax() - 1, 'order_quantity']
sales.loc[sales['order_value'].idxmax(), 'order_value'] = sales.loc[sales['order_value'].idxmax() - 1, 'order_value']
# Group by Week
sales['date'] = sales['date'].dt.to_period('W').apply(lambda r: r.start_time)
sales = sales.groupby('date').sum().reset_index()
print(sales.head())


auto_model = auto_arima(sales['order_quantity'])
print(auto_model.summary())


# Split Training and Testing
df_len = sales.shape[0]
test_size = .2
test_size = round(test_size * df_len)
train = sales[:-test_size]
test = sales[-test_size:]


# 4,1,3 from auto arima
ARIMA_MODEL(
    train_set=train, 
    test_set=test, 
    series_col='order_quantity', 
    date_col='date', 
    p_d_q=(4,1,3)
)

# SARIMA Model
ARIMA_MODEL(
    train_set=train, 
    test_set=test, 
    series_col='order_quantity', 
    date_col='date', 
    p_d_q=(2,1,2), 
    P_D_Q_S=(1,0,1,52)
)


# Lets try monthly... 
sales['month-yr'] = sales['date'].dt.strftime("%Y-%m")
msales = sales.groupby('month-yr').agg({
    'order_quantity':'sum'
}).reset_index()

# Plot Monthly Sales
plt.figure(figsize=(12, 6))
plt.plot(msales['month-yr'], msales['order_quantity'], marker='o')
plt.title('Monthly Sales Quantity')
plt.xlabel('Month-Year')
plt.ylabel('Order Quantity')
plt.xticks(rotation=45)
plt.show()

auto_model = auto_arima(msales['order_quantity'])
print(auto_model.summary())

# Train Test Split
len_df = msales.shape[0]
test_size = round(len_df * .2)
test = msales[-test_size:]
train = msales[:-test_size]

# Fit Model
# Auto Rec (1,1,1)
ARIMA_MODEL(
    train_set=train,
    test_set=test, 
    series_col='order_quantity', 
    date_col='month-yr', 
    p_d_q=(1,1,1)
)

# Auto SARIMA...
auto_model = auto_arima(msales['order_quantity'], 
           seasonal=True, 
           m=12, 
           trace=True)
print(auto_model.summary())

# Auto Says 1,1,1,0,0,0,12
ARIMA_MODEL(
    train_set=train,
    test_set=test, 
    series_col='order_quantity', 
    date_col='month-yr', 
    p_d_q=(1,1,1), 
    P_D_Q_S=(0,0,0,12)
)

# Testing Random Shit
ARIMA_MODEL(
    train_set=train,
    test_set=test, 
    series_col='order_quantity', 
    date_col='month-yr', 
    p_d_q=(1,0,1),
    P_D_Q_S=(1,1,1,12) 
)

# Log Transform
msales['log_order_quantity'] = np.log(msales['order_quantity'])
train = msales[:-test_size]
test = msales[-test_size:]

auto_model = auto_arima(msales['log_order_quantity'],
              seasonal=True, 
              m=12, 
              trace=True)
print(auto_model.summary())

# auto Says 0,1,1 x 2,0,0,12
ARIMA_MODEL(
    train_set=train,
    test_set=test, 
    series_col='log_order_quantity', 
    date_col='month-yr', 
    p_d_q=(1,0,1),
    P_D_Q_S=(1,1,1,12) 
)
