#!/usr/bin/env python3

import numba
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

#%% Import data ----------------------------------------------------------------

bikes = pd.read_csv(
  'data/divvy_bikes_chicago_weather_2013-2017_transformed.csv',
  parse_dates = ['datetime'],
  infer_datetime_format = True
)

#%% Annotate data --------------------------------------------------------------

# Get length of contiguous sequence following each row
# A contiguous sequence is a time-sequence of rows in the same station
switches = np.where(
  (bikes['datetime'] - dt.timedelta(hours = 1) != bikes['datetime'].shift(1)) |
  (bikes['x_coordinate'] != bikes['x_coordinate'].shift(1)) |
  (bikes['y_coordinate'] != bikes['y_coordinate'].shift(1)) |
  (bikes['z_coordinate'] != bikes['z_coordinate'].shift(1)),
  1,
  0
) # [1,0,0,0,1,0,0,1,0]
indices = np.where(np.append(switches, 1))[0] # [0,4,7,9]
lengths = np.diff(indices) # [4,3,2]
offsets = np.cumsum(lengths - 1).repeat(lengths) # [3,3,3,3,5,5,5,6,6]
lengths = offsets - np.cumsum(1 - switches) + 1 # [4,3,2,1,3,2,1,2,1]
bikes['contiguous_length'] = lengths

# Mark seasons
bikes['season'] = pd.Series(np.select(
  [
    bikes['datetime'].dt.month.isin([12, 1, 2]),
    bikes['datetime'].dt.month.isin([3, 4, 5]),
    bikes['datetime'].dt.month.isin([6, 7, 8]),
    bikes['datetime'].dt.month.isin([9, 10, 11]),
  ],
  [
    'winter',
    'spring',
    'summer',
    'autumn'
  ]
)).astype('category')

# Mark precipitation for each 24h sequence
bikes['precipitation'] = pd.Series(np.where(
  (bikes['rain_1h'].rolling(24).max().shift(-24+1) > 0) |
  (bikes['snow_1h'].rolling(24).max().shift(-24+1) > 0),
  'wet',
  'dry'
)).astype('category')

bikes['precipitation'] = pd.Series(np.where(
  bikes['contiguous_length'] < 24,
  pd.NA,
  bikes['precipitation']
)).astype('category')

# Mark bike availability variation for each 24h sequence
bikes['variation'] = (
  bikes['available_bikes'].rolling(24).max().shift(-24+1) -
  bikes['available_bikes'].rolling(24).min().shift(-24+1)
)

bikes['variation'] = np.where(
  bikes['contiguous_length'] < 24,
  np.nan,
  bikes['variation']
).astype('float32')

#%% Plot variation distribution ------------------------------------------------

sns.histplot(
  x = bikes[bikes['contiguous_length'] >= 24]['variation'],
  bins = 10
)

#%% Plot seasons and precipitation distributions -------------------------------

sns.countplot(
  data = bikes[bikes['contiguous_length'] >= 24],
  x = 'season',
  hue = 'precipitation'
)

#%% Plot variation by season and precipitation ---------------------------------

sns.violinplot(
  data = bikes[bikes['contiguous_length'] >= 24].sample(10000),
  x = 'variation',
  y = 'season',
  hue = 'precipitation',
  inner = 'quartile',
  scale = 'width',
  bw = 0.2,
  cut = 0
)

#%% Select a subset of sequences -----------------------------------------------

def sample_start_indices(dataframe, sample_count, sequence_length):

  # Find groups of contiguous sequences
  group_switches = np.where(
    (dataframe['datetime'] - dt.timedelta(hours = 1) != dataframe['datetime'].shift(1)) |
    (dataframe['x_coordinate'] != dataframe['x_coordinate'].shift(1)) |
    (dataframe['y_coordinate'] != dataframe['y_coordinate'].shift(1)) |
    (dataframe['z_coordinate'] != dataframe['z_coordinate'].shift(1)),
    1,
    0
  )

  # Get start index and length of each group
  group_indices = np.where(np.append(group_switches, 1))[0]
  group_lengths = np.diff(group_indices)
  group_indices = group_indices[:-1]

  # Determine number of samples to select in each group
  sample_counts = np.ceil(sample_count * (group_lengths / len(group_switches)))

  # Sample sequence start indices from each group
  start_indices = np.array([], dtype = int)
  for count, limit, offset in zip(sample_counts, group_lengths, group_indices):
    count = int(min(limit // sequence_length, count))
    limit = int(limit - count * (sequence_length - 1))
    indices = np.random.choice(limit, count, replace = False)
    indices = np.arange(count) * (sequence_length - 1) + np.sort(indices)
    start_indices = np.concatenate((start_indices, offset + indices))

  # We might have selected too many total samples because of np.ceil
  if len(start_indices) > sample_count:
    selection = np.random.choice(len(start_indices), sample_count, replace = False)
    start_indices = start_indices[np.sort(selection)]

  # Return sampled sequence indices
  return dataframe.iloc[start_indices].index.to_numpy()

# Select subset indices
subset_indices = []
for i in range(10):

  candidate_indices = bikes[
    (bikes['variation'] >= i/10) &
    (bikes['variation'] < ((i+1)/10 if i < 9 else 1.1)) &
    (bikes['contiguous_length'] >= 24)
  ].index.to_numpy()

  # Add sequence indices after each sequence start index
  candidate_indices = candidate_indices + np.expand_dims(np.arange(24), 1)
  candidate_indices = np.unique(candidate_indices.transpose().flatten())

  subset_indices += [
    sample_start_indices(
      bikes.loc[candidate_indices],
      sample_count = int(500000 * [0.27, 0.14, 0.09, 0.21, 0.07, 0.19, 0.09, 1, 1, 1][i]),
      sequence_length = 24
    )
  ]

subset_indices = np.sort(np.concatenate(subset_indices))

# Add sequence indices after each sequence start index
subset_indices = subset_indices + np.expand_dims(np.arange(24), 1)
subset_indices = np.unique(subset_indices.transpose().flatten())

# Select subset
subset = bikes.loc[subset_indices]

# Get length of contiguous sequence following each row
# A contiguous sequence is a time-sequence of rows in the same station
switches = np.where(
  (subset['datetime'] - dt.timedelta(hours = 1) != subset['datetime'].shift(1)) |
  (subset['x_coordinate'] != subset['x_coordinate'].shift(1)) |
  (subset['y_coordinate'] != subset['y_coordinate'].shift(1)) |
  (subset['z_coordinate'] != subset['z_coordinate'].shift(1)),
  1,
  0
) # [1,0,0,0,1,0,0,1,0]
indices = np.where(np.append(switches, 1))[0] # [0,4,7,9]
lengths = np.diff(indices) # [4,3,2]
offsets = np.cumsum(lengths - 1).repeat(lengths) # [3,3,3,3,5,5,5,6,6]
lengths = offsets - np.cumsum(1 - switches) + 1 # [4,3,2,1,3,2,1,2,1]
subset['contiguous_length'] = lengths

#%% Plot variation distribution ------------------------------------------------

sns.histplot(
  x = subset[subset['contiguous_length'] >= 24]['variation'],
  bins = 10
)

#%% Plot seasons and precipitation distributions -------------------------------

sns.countplot(
  data = subset[subset['contiguous_length'] >= 24],
  x = 'season',
  hue = 'precipitation'
)

#%% Plot variation by season and precipitation ---------------------------------

sns.violinplot(
  data = subset[subset['contiguous_length'] >= 24].sample(10000),
  x = 'variation',
  y = 'season',
  hue = 'precipitation',
  inner = 'quartile',
  scale = 'width',
  bw = 0.2,
  cut = 0
)

#%% Output subset --------------------------------------------------------------

subset[[
  'datetime',
  'contiguous_length',
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
  'data/divvy_bikes_chicago_weather_2013-2017_sampled.csv',
  index = False
)

print(subset.info())
