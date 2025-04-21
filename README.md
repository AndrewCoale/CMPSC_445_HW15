# CMPSC_445_HW15

## Results:
Mean Squared Error: 0.000316  
Predicted Opening Price on April 26: $406.11  

## Discussion:
Straightforward preprocessing, just sorted the dataset by date, normalized the data with MinMaxScaler, and isolated the last 20 days of data. Then it was as simple as training an MLPRegressor on that data, evaluating its mean squared error, and using it to predict the opening stock price value on April 26. Based on the MSE, it seems to have worked well, but we won't really know for 5 days.
