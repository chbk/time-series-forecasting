#!/usr/bin/env python3

import re
import glob
import haversine
import numpy as np
import pandas as pd

#%% Parse weather --------------------------------------------------------------

weather = pd.read_csv('data/weather/chicago.csv').rename(
  columns = {
    'dt_iso': 'datetime',
    'temp': 'temperature',
    'feels_like': 'temperature_felt',
    'temp_min': 'temperature_min',
    'temp_max': 'temperature_max',
    'wind_deg': 'wind_direction',
    'clouds_all': 'cloudiness',
    'weather_id': 'code_id',
    'weather_main': 'code_main',
    'weather_description': 'code_description'
  }
).drop(
  [
    'dt',
    'timezone',
    'city_name',
    'lat',
    'lon',
    'sea_level',
    'grnd_level',
    'weather_icon'
  ],
  axis = 1
)

weather['datetime'] = pd.to_datetime(
  pd.to_datetime(
    weather['datetime'],
    format = '%Y-%m-%d %H:%M:%S %z UTC'
  ).dt.strftime('%Y-%m-%d %H:%M:%S')
)

weather = weather.drop_duplicates(subset = 'datetime')

weather = weather.set_index('datetime').sort_values('datetime')

weather = weather.loc['2013-01-01':'2017-12-31']

weather['humidity'] = weather['humidity'].fillna(0)
weather['cloudiness'] = weather['cloudiness'].fillna(0)
weather['wind_speed'] = weather['wind_speed'].fillna(0)
weather['rain_1h'] = weather['rain_1h'].fillna(0)
weather['rain_3h'] = weather['rain_3h'].fillna(0)
weather['snow_1h'] = weather['snow_1h'].fillna(0)
weather['snow_3h'] = weather['snow_3h'].fillna(0)

#%% Output weather -------------------------------------------------------------

weather.to_csv('data/chicago_weather_2013-2017.csv')

print(weather.info())

#%% Parse stations -------------------------------------------------------------

stations = pd.DataFrame()

for file in sorted(glob.glob('data/stations/*.csv')):

  df = pd.read_csv(file).rename(
    columns = {
      'id': 'station_id',
      'dpcapacity': 'capacity'
    }
  )

  df['snapshot_date'] = pd.to_datetime(
    re.sub(r'^.+/([-\d]+)\.csv$', '\\1', file),
    format = '%Y-%m-%d'
  )

  df = df[[
    'station_id',
    'snapshot_date',
    'name',
    'latitude',
    'longitude',
    'capacity'
  ]]

  stations = pd.concat((stations, df))

stations['latitude'] = np.around(stations['latitude'], decimals = 6)
stations['longitude'] = np.around(stations['longitude'], decimals = 6)

stations = stations.sort_values(['station_id', 'snapshot_date'])

stations = stations.reset_index(drop = True)
stations.index.name = 'snapshot_id'

#%% Check stations consistency -------------------------------------------------

pairs = pd.concat(
  (
    stations.iloc[
      np.repeat(np.arange(len(stations)), len(stations))
    ].reset_index(),
    stations.iloc[
      np.tile(np.arange(len(stations)), len(stations))
    ].reset_index().add_prefix('pair_')
  ),
  axis = 1
)

pairs = pairs[pairs['station_id'] <= pairs['pair_station_id']]

coordinates = np.column_stack([
  pairs['latitude'],
  pairs['longitude']
])
pair_coordinates = np.column_stack([
  pairs['pair_latitude'],
  pairs['pair_longitude']
])
pairs['pair_distance'] = haversine.haversine_vector(
  coordinates,
  pair_coordinates,
  unit = 'm'
)

# Observations:

#  - Between two snapshot dates, a same station id can have different names
print(len(pairs[
  (pairs['snapshot_date'] != pairs['pair_snapshot_date']) &
  (pairs['station_id'] == pairs['pair_station_id']) &
  (pairs['name'] != pairs['pair_name'])
]))

#  - Between two snapshot dates, a same station id can have different capacities
print(len(pairs[
  (pairs['snapshot_date'] != pairs['pair_snapshot_date']) &
  (pairs['station_id'] == pairs['pair_station_id']) &
  (pairs['capacity'] != pairs['pair_capacity'])
]))

#  - Between two snapshot dates, a same station id can be over 90 meters apart
print(max(pairs[
  (pairs['snapshot_date'] != pairs['pair_snapshot_date']) &
  (pairs['station_id'] == pairs['pair_station_id'])
]['pair_distance']))

#  - At any given snapshot date, all stations are more than 90 meters apart
print(min(pairs[
  (pairs['snapshot_date'] == pairs['pair_snapshot_date']) &
  (pairs['station_id'] != pairs['pair_station_id'])
]['pair_distance']))

#  - At any given snapshot date, all stations have a unique station id
print(len(pairs[
  (pairs['snapshot_date'] == pairs['pair_snapshot_date']) &
  (pairs['station_id'] == pairs['pair_station_id']) &
  (pairs['pair_distance'] != 0)
]))

# Assumptions:
#  - Stations at most 90 meters apart with the same station id are the same
#    relocated station
#  - Stations more than 90 meters apart with the same station id are different
#    stations with a repurposed station id
#  - Stations of different capacities with the same station id are
#    enlarged/shrunk stations that we consider as different stations

# Conclusion:
#  - Every station is given proxy coordinates, which are the average coordinates
#    of the same station over all snapshot dates

#%% Average station coordinates ------------------------------------------------

stations = pd.concat(
  (
    stations,
    stations.add_prefix('previous_').shift(1)
  ),
  axis = 1
)

coordinates = np.column_stack([
  stations['latitude'],
  stations['longitude']
])
previous_coordinates = np.column_stack([
  stations['previous_latitude'],
  stations['previous_longitude']
])
stations['previous_distance'] = haversine.haversine_vector(
  coordinates,
  previous_coordinates,
  unit = 'm'
)

stations['proxy_id'] = (
  (stations['station_id'] != stations['previous_station_id']) |
  (stations['capacity'] != stations['previous_capacity']) |
  (stations['previous_distance'] > 90)
).cumsum() - 1

stations['proxy_latitude'] = np.around(
  stations.groupby('proxy_id')['latitude'].transform('mean'),
  decimals = 6
)
stations['proxy_longitude'] = np.around(
  stations.groupby('proxy_id')['longitude'].transform('mean'),
  decimals = 6
)

coordinates = np.column_stack([
  stations['latitude'],
  stations['longitude']
])
proxy_coordinates = np.column_stack([
  stations['proxy_latitude'],
  stations['proxy_longitude']
])
stations['proxy_distance'] = np.around(
  haversine.haversine_vector(coordinates, proxy_coordinates, 'm'),
  decimals = 2
)

stations = stations[[
  'station_id',
  'snapshot_date',
  'name',
  'capacity',
  'latitude',
  'longitude',
  'proxy_id',
  'proxy_latitude',
  'proxy_longitude',
  'proxy_distance'
]]

#%% Output stations ------------------------------------------------------------

stations.to_csv('data/divvy_stations_2013-2017.csv')

print(stations.info())

#%% Parse trips ----------------------------------------------------------------

trips = pd.DataFrame()

for file in sorted(glob.glob('data/trips/*.csv')):

  df = pd.read_csv(
    file,
    dtype = {
      'trip_id': int,
      'starttime': str,
      'start_time': str,
      'stoptime': str,
      'end_time': str,
      'bikeid': int,
      'tripduration': int,
      'from_station_id': int,
      'from_station_name': str,
      'to_station_id': int,
      'to_station_name': str,
      'usertype': str,
      'gender': str,
      'birthday': str,
      'birthyear': str
    },
    index_col = 'trip_id'
  ).rename(
    columns = {
      'starttime': 'start_datetime',
      'start_time': 'start_datetime',
      'stoptime': 'stop_datetime',
      'end_time': 'stop_datetime',
      'tripduration': 'duration',
      'from_station_id': 'start_station_id',
      'from_station_name': 'start_station_name',
      'to_station_id': 'stop_station_id',
      'to_station_name': 'stop_station_name',
      'bikeid': 'bike_id',
      'usertype': 'user_type',
      'gender': 'user_gender',
      'birthyear': 'user_birth_year',
      'birthday': 'user_birth_year',
    }
  ).drop(
    [
      'duration',
      'bike_id',
      'user_type',
      'user_gender',
      'user_birth_year'
    ],
    axis = 1
  )

  df['start_datetime'] = pd.to_datetime(
    df['start_datetime'],
    infer_datetime_format = True
  )
  df['stop_datetime'] = pd.to_datetime(
    df['stop_datetime'],
    infer_datetime_format = True
  )

  trips = pd.concat((trips, df))

trips = trips.sort_values(['trip_id', 'start_datetime', 'stop_datetime'])

trips = trips[~trips.index.duplicated()]

#%% Output trips ---------------------------------------------------------------

trips.to_csv('data/divvy_trips_2013-2017.csv')

print(trips.info())
