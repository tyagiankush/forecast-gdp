import requests
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from pmdarima.arima import auto_arima
import argparse


class ForecastModeler:
    def __init__(self, forecast_period_val: int):
        self.forecast_period = forecast_period_val

    def arima_forecast(self, series: pd.DataFrame) -> pd.DataFrame:
        model = auto_arima(series, seasonal=False, suppress_warnings=True)
        forecast_data = model.predict(n_periods=self.forecast_period)
        return forecast_data

    def prophet_forecast(self, series: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame({'ds': list(series.index), 'y': series.value})
        m = Prophet(seasonality_mode='multiplicative')
        m.fit(df)
        future = m.make_future_dataframe(periods=self.forecast_period, freq='Y')
        forecast_data = m.predict(future)['yhat'][-1 * self.forecast_period:].values
        return forecast_data

    def get_mean_forecast(self, series: pd.DataFrame):
        arima = self.arima_forecast(series)
        prophet = self.prophet_forecast(series)
        print_if_debug_enabled(arima)
        print_if_debug_enabled(prophet)
        ensemble = np.mean([arima, prophet], axis=0)
        return ensemble


class WorldBankForecaster:
    def __init__(self, country_code_val: str, series_code_val: str):
        self.series_code = series_code_val
        self.country_code = country_code_val
        self.url = f"https://api.worldbank.org/v2/country/{self.country_code}/indicator/{self.series_code}"
        self.params = {
            "per_page": "1000",
            "format": "json"
        }
        self.DATE, self.YEAR, self.HISTORIC = 'date', 'year', 'historic_gdp'

    @staticmethod
    def process_response(response, country_code_val: str):
        if len(response.json()) < 2:
            raise ValueError("Empty response received for country code '{}'".format(country_code_val))
        print_if_debug_enabled(response.json()[1])
        return response.json()[1]

    @staticmethod
    def convert_raw_data_to_dataframe(raw_data) -> pd.DataFrame:
        return pd.json_normalize(raw_data)

    @staticmethod
    def generate_output_json(country_code_val: str, series_code_val: str, result_json, forecast_years):
        output = {
            "series_code": series_code_val,
            "country_code": country_code_val,
            "data": result_json,
            "forecast_years": forecast_years
        }
        with open("./target/forecast_{}.json".format(country_code_val), "w") as f:
            json.dump(output, f)

    def download_data(self) -> pd.DataFrame:
        response = requests.get(self.url, params=self.params)
        raw_data = WorldBankForecaster.process_response(response, self.country_code)
        df = WorldBankForecaster.convert_raw_data_to_dataframe(raw_data)
        return df

    def clean_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[[self.DATE, 'value']].dropna().sort_values(by=self.DATE)
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        df = df.set_index(self.DATE)
        print_if_debug_enabled(df)
        return df

    def concat_data(self, forecast_val, clean_df: pd.DataFrame):
        end_year = clean_df.index[-1].year
        forecast_years = list(range(end_year + 1, 2031))
        forecast_index = pd.date_range(start=f"{end_year + 1}-01-01", end=f"{end_year + 10}-01-01", freq='Y')
        forecast_data = pd.Series(data=forecast_val, index=forecast_index)
        result_df = pd.concat([clean_df, forecast_data])
        return result_df, forecast_years

    def post_process_and_save_result(self, result: pd.DataFrame, forecast_years_val: list):
        result_df = result.reset_index().rename(
            columns={'index': self.DATE, 'value': self.HISTORIC, 0: 'forecast'}).fillna(0)
        result_df[self.YEAR] = result_df[self.DATE].dt.year
        result_df = result_df.drop(self.DATE, axis=1)
        result_df = result_df[[self.YEAR, self.HISTORIC, 'forecast']]
        result_json = result_df.to_dict('records')
        print_if_debug_enabled(result_df)
        WorldBankForecaster.generate_output_json(self.country_code, self.series_code, result_json, forecast_years_val)


def main():
    try:
        # Create objects for the above classes
        forecast_modeler = ForecastModeler(forecast_period)
        world_bank_forecaster = WorldBankForecaster(country_code, series_code)

        # Download data from WB API
        df = world_bank_forecaster.download_data()
        # Process the raw response and return a clean DF
        data = world_bank_forecaster.clean_raw_dataframe(df)
        # Forecast using Arima and Prophet models and return a mean of the models
        forecast = forecast_modeler.get_mean_forecast(data)
        # Concatenate forecast and historic data
        complete_df, forecast_years = world_bank_forecaster.concat_data(forecast, data)
        # Post process and save the result in a Json file
        world_bank_forecaster.post_process_and_save_result(complete_df, forecast_years)
    except ValueError as ve:
        print(ve)


def parse_args():
    parser = argparse.ArgumentParser(description='World Bank Forecaster Arg Parser')
    parser.add_argument('-c', '--code', type=str, default='afg', help='country code, default AFG')
    parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug flag, default False')
    return parser.parse_args()


def print_if_debug_enabled(val):
    if debug_enabled:
        print(val)


if __name__ == "__main__":
    args = parse_args()
    debug_enabled = args.debug
    country_code = args.code
    forecast_period = 9
    series_code = "NY.GDP.MKTP.CN"
    main()
