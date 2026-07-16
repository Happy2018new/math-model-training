import openmeteo_requests

from pathlib import Path

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 48.86,
	"longitude": 2.36,
	"start_date": "2016-05-01",
	"end_date": "2026-05-01",
	"daily": ["wind_direction_10m_dominant", "wind_speed_10m_mean", "weather_code"],
	"hourly": ["wind_direction_100m", "wind_speed_100m"],
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_wind_direction_100m = hourly.Variables(0).ValuesAsNumpy()
hourly_wind_speed_100m = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)
}

hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
hourly_data["wind_speed_100m"] = hourly_wind_speed_100m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_wind_direction_10m_dominant = daily.Variables(0).ValuesAsNumpy()
daily_wind_speed_10m_mean = daily.Variables(1).ValuesAsNumpy()
daily_weather_code = daily.Variables(2).ValuesAsNumpy()

daily_data = {
	"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)
}

daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
daily_data["weather_code"] = daily_weather_code

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)

output_dir = Path("weather_output")
output_dir.mkdir(exist_ok=True)

hourly_csv_path = output_dir / "paris_hourly_weather_2016-05-01_to_2026-05-01.csv"
daily_csv_path = output_dir / "paris_daily_weather_2016-05-01_to_2026-05-01.csv"

hourly_dataframe.to_csv(hourly_csv_path, index=False, encoding="utf-8-sig")
daily_dataframe.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")

print(f"\nSaved hourly CSV: {hourly_csv_path}")
print(f"Saved daily CSV: {daily_csv_path}")
