# forecast-gdp

## Forecasts GDP till 2030 for countries in the World Bank DB using ARIMA and Prophet models

To execute, use below command - 

{YOUR_PYTHON_PATH}/python3 forecast_gdp.py -c {country code/ default is AFG} [-d True (For printing in console/ default is False)]

eg - 
  - python3 forecast_gdp.py -c usa
  - For Printing data on console: python3 forecast_gdp.py -c usa -d True
