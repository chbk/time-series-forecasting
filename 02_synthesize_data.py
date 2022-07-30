#!/usr/bin/env python3

import numba
import numpy as np
import pandas as pd

# Import stations
stations = pd.read_csv(
  'data/divvy_stations_2013-2017.csv',
  usecols = [
    'snapshot_id',
    'capacity',
    'proxy_id',
    'proxy_latitude',
    'proxy_longitude'
  ]
)

# Import trips
trips = pd.read_csv(
  'data/divvy_trips_2013-2017.csv',
  parse_dates = ['start_datetime', 'stop_datetime'],
  infer_datetime_format = True,
  usecols = [
    'start_datetime',
    'stop_datetime',
    'start_station_snapshot_id',
    'stop_station_snapshot_id'
  ]
)

# Round datetimes to next hour
trips['start_datetime'] = trips['start_datetime'].dt.ceil('h')
trips['stop_datetime'] = trips['stop_datetime'].dt.ceil('h')

# Each trip changes the number of bikes in two stations
bikes = pd.concat((trips, trips))

arrivals = bikes.index.duplicated()

# Set +1 bike for arrivals, -1 bike for departures
bikes['variation'] = np.where(arrivals, 1, -1)

# Keep stop datetime for arrivals, start datetime for departures
bikes['datetime'] = bikes['stop_datetime'].where(
  arrivals,
  bikes['start_datetime']
)

# Keep stop snapshot id for arrivals, start snapshot id for departures
bikes['snapshot_id'] = bikes['stop_station_snapshot_id'].where(
  arrivals,
  bikes['start_station_snapshot_id']
)

# Sort by station and datetime
bikes = bikes.merge(
  stations,
  on = 'snapshot_id'
).drop(
  [
    'start_datetime',
    'stop_datetime',
    'start_station_snapshot_id',
    'stop_station_snapshot_id',
    'snapshot_id'
  ],
  axis = 1
).sort_values(
  ['proxy_id', 'datetime']
)

# Delete stations with 0 capacity
bikes = bikes[bikes['capacity'] > 0]

# Sum variations occuring at the same datetime
bikes = bikes.groupby(
  ['proxy_id', 'proxy_latitude', 'proxy_longitude', 'capacity', 'datetime'],
  as_index = False
).agg({
  'variation': sum
}).set_index(
  ['proxy_id', 'datetime']
)

# Infer the number of available bikes based on variation and capacity
@numba.njit
def bounded_cumulative_sum(array, min = np.nan, max = np.nan, start = 0):
  result = np.zeros(array.size)
  previous = start
  for i in range(0, array.size):
    result[i] = np.fmax(min, np.fmin(max, previous + array[i]))
    previous = result[i]
  return result

bikes['available_bikes'] = bikes.groupby('proxy_id').apply(
  lambda group: bounded_cumulative_sum(
    group['variation'].to_numpy(),
    min = 0,
    max = group['capacity'].iloc[0],
    start = group['capacity'].iloc[0] // 2
  )
).explode().astype(int).values

bikes = bikes.drop('variation', axis = 1)

# Remove first 200 changes of each station for bike availability calibration
bikes = bikes.groupby('proxy_id').apply(
  lambda group: group.reset_index(
      'proxy_id',
      drop = True
    ).iloc[200:]
)

# Fill missing datetimes with preceding values
bikes = bikes.groupby('proxy_id').apply(
  lambda group: group.reset_index(
      ['proxy_id'],
      drop = True
    ).reindex(
      pd.date_range(
        start = group.index.get_level_values('datetime').min(),
        end = group.index.get_level_values('datetime').max(),
        freq = '1H'
      ),
      method = 'pad'
    ).rename_axis(index = 'datetime')
)

# Output bikes
bikes.to_csv('data/divvy_bikes_2013-2017.csv')

print(bikes.info())
