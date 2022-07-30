#!/usr/bin/env python3

import haversine
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import plotly.express as px

#%% Import data ----------------------------------------------------------------

bikes = pd.read_csv(
  'data/divvy_bikes_2013-2017.csv',
  parse_dates = ['datetime'],
  infer_datetime_format = True
)

weather = pd.read_csv(
  'data/chicago_weather_2013-2017.csv',
  parse_dates = ['datetime'],
  infer_datetime_format = True,
  usecols = [
    'datetime',
    'temperature',
    'humidity',
    'cloudiness',
    'wind_speed',
    'rain_1h',
    'snow_1h'
  ]
)

#%% Scale available bikes ------------------------------------------------------

bikes['available_bikes'] = (bikes['available_bikes'] / bikes['capacity']).round(6)

#%% Plot available bikes distribution ------------------------------------------

sns.stripplot(
  data = bikes[['available_bikes']].sample(10000),
  size = 0.5,
  orient = 'h'
)

#%% Normalize station coordinates ----------------------------------------------

locations = bikes[
  ['proxy_latitude', 'proxy_longitude']
].drop_duplicates().reset_index(drop = True)

# Project latitudes and longitudes to Cartesian coordinates
locations['x_coordinate'] = (
  np.cos(locations['proxy_latitude']) *
  np.cos(locations['proxy_longitude'])
)
locations['y_coordinate'] = (
  np.cos(locations['proxy_latitude']) *
  np.sin(locations['proxy_longitude'])
)
locations['z_coordinate'] = - np.sin(locations['proxy_latitude'])

# Add distance to dowtown
madison_state_intersection = (41.882045, -87.627823)

locations['downtown_distance'] = haversine.haversine_vector(
  [madison_state_intersection] * len(locations),
  locations[['proxy_latitude','proxy_longitude']],
  haversine.Unit.KILOMETERS
)

# Z-score normalize coordinates
locations['x_coordinate'] = (locations['x_coordinate'] + 0.47) / 0.05
locations['y_coordinate'] = (locations['y_coordinate'] + 0.15) / 0.03
locations['z_coordinate'] = (locations['z_coordinate'] - 0.87) / 0.03

# Scale distance
locations['downtown_distance'] /= 20

bikes = bikes.merge(
  locations.round(6),
  on = ['proxy_latitude', 'proxy_longitude'],
  how = 'left'
)

#%% Plot station coordinates ---------------------------------------------------

px.scatter_3d(
  locations,
  x = 'x_coordinate',
  y = 'y_coordinate',
  z = 'z_coordinate',
  color = 'downtown_distance',
).update_traces(marker_size = 2)

#%% Transform datetimes --------------------------------------------------------

datetimes = bikes[['datetime']].drop_duplicates().reset_index(drop = True)

# Transform hours and days
day_hour = datetimes['datetime'].dt.hour
datetimes['sin_day_hour'] = np.sin(2 * np.pi * day_hour / 24 - np.pi)
datetimes['cos_day_hour'] = np.cos(2 * np.pi * day_hour / 24 - np.pi)

year_day = datetimes['datetime'].dt.dayofyear
datetimes['sin_year_day'] = np.sin(2 * np.pi * year_day / 365 - np.pi)
datetimes['cos_year_day'] = np.cos(2 * np.pi * year_day / 365 - np.pi)

week_hour = 24 * datetimes['datetime'].dt.dayofweek + datetimes['datetime'].dt.hour
datetimes['sin_week_hour'] = np.sin(2 * np.pi * week_hour / (24 * 7) - np.pi)
datetimes['cos_week_hour'] = np.cos(2 * np.pi * week_hour / (24 * 7) - np.pi)

# Add a workday feature
holidays = [
  (dt.datetime(year, 1, 1) + offset).date()
  for year in range(
    datetimes['datetime'].min().year,
    datetimes['datetime'].max().year + 1
  )
  for offset in [
    # New year on 1st of January
    pd.tseries.offsets.DateOffset(months = 0, days = 0),
    # MLK day on 3rd Monday of January
    pd.tseries.offsets.DateOffset(months = 0, weeks = 2, weekday = 0),
    # President day on 3rd Monday of February
    pd.tseries.offsets.DateOffset(months = 1, weeks = 2, weekday = 0),
    # Memorial day on last Monday of May
    pd.tseries.offsets.LastWeekOfMonth(5, weekday = 0),
    # Independence day on 4th of July
    pd.tseries.offsets.DateOffset(months = 6, days = 3),
    # Labor day on 1st Monday of September
    pd.tseries.offsets.DateOffset(months = 8, weeks = 0, weekday = 0),
    # Columbus day on 2nd Monday of October
    pd.tseries.offsets.DateOffset(months = 9, weeks = 1, weekday = 0),
    # Veterans day on 11th of November
    pd.tseries.offsets.DateOffset(months = 10, days = 10),
    # Thanksgiving on 4th Thursday of November
    pd.tseries.offsets.DateOffset(months = 10, weeks = 3, weekday = 3),
    # Christmas on 25th of December
    pd.tseries.offsets.DateOffset(months = 11, days = 24)
  ]
]

datetimes['workday'] = np.where(
  datetimes['datetime'].dt.date.isin(holidays) |
  datetimes['datetime'].dt.dayofweek.isin([5, 6]),
  0,
  1
)

bikes = bikes.merge(datetimes.round(6), on = 'datetime', how = 'left')

#%% Plot time transformations --------------------------------------------------

x = np.arange(0, 24, 0.1)
sin_day_hour = np.sin(2 * np.pi * x / 24 - np.pi)
cos_day_hour = np.cos(2 * np.pi * x / 24 - np.pi)
sns.lineplot(
  data = pd.DataFrame({
    'hour': x,
    'sin_day_hour': sin_day_hour,
    'cos_day_hour': cos_day_hour,
    'day_hour': (np.arctan2(sin_day_hour, cos_day_hour) + np.pi) / (2 * np.pi)
  }).set_index('hour'),
)

#%% Scale weather --------------------------------------------------------------

weather['temperature'] += 25
weather['temperature'] /= 60
weather['humidity'] /= 100
weather['cloudiness'] /= 100
weather['wind_speed'] /= 20
weather['rain_1h'] /= 30
weather['snow_1h'] /= 10

bikes = bikes.merge(weather.round(6), on = 'datetime', how = 'left')

#%% Plot weather distributions -------------------------------------------------

sns.violinplot(
  data = weather[[
    'temperature',
    'humidity',
    'cloudiness',
    'wind_speed',
    'rain_1h',
    'snow_1h'
  ]],
  inner = 'quartile',
  scale = 'width',
  orient = 'h',
  cut = 0
)

#%% Output transformed data ----------------------------------------------------

bikes[[
  'datetime',
  'available_bikes',
  'x_coordinate',
  'y_coordinate',
  'z_coordinate',
  'downtown_distance',
  'sin_year_day',
  'cos_year_day',
  'sin_week_hour',
  'cos_week_hour',
  'sin_day_hour',
  'cos_day_hour',
  'workday',
  'temperature',
  'humidity',
  'cloudiness',
  'wind_speed',
  'rain_1h',
  'snow_1h'
]].to_csv(
  'data/divvy_bikes_chicago_weather_2013-2017_transformed.csv',
  index = False
)

print(bikes.info())
