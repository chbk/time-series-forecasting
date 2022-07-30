#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import contextily as ctx
import matplotlib as mpl

# Create plot directories
os.makedirs('plots/stations', exist_ok = True)
os.makedirs('plots/dates', exist_ok = True)

# Configure matplotlib
mpl.pyplot.subplots()
mpl.rc('font', family = 'Roboto Mono')
mpl.rc('xtick', bottom = False)
mpl.rc('ytick', left = False)
mpl.rc('xtick.major', bottom = False)
mpl.rc('ytick.major', left = False)
mpl.rc('axes.spines', left = False)
mpl.rc('axes.spines', top = False)
mpl.rc('axes.spines', bottom = False)
mpl.rc('axes.spines', right = False)
mpl.rc('axes', facecolor = (1,1,1,0))
mpl.rc('figure', facecolor = (1,1,1,0))
mpl.rc('figure', dpi = 300)
mpl.rc('figure.constrained_layout', use = True)
mpl.rc('figure.constrained_layout', h_pad = 0)
mpl.rc('figure.constrained_layout', w_pad = 0)
mpl.rc('figure.constrained_layout', hspace = 0)
mpl.rc('figure.constrained_layout', wspace = 0)
mpl.pyplot.close()

# Import bikes
bikes = pd.read_csv(
  'data/divvy_bikes_2013-2017.csv',
  parse_dates = ['datetime'],
  infer_datetime_format = True
)

# Get percentage of available bikes
bikes['available_bikes_percent'] = (
  100 * bikes['available_bikes'] / bikes['capacity']
)

# Convert latitudes and longitudes to Web Mercator coordinates
stations = bikes[
  ['proxy_latitude', 'proxy_longitude']
].drop_duplicates()

stations['geometry'] = gpd.points_from_xy(
  x = stations['proxy_longitude'],
  y = stations['proxy_latitude'],
  crs ='EPSG:4326'
).to_crs(epsg = 3857)

bikes = gpd.GeoDataFrame(
  bikes.merge(stations, on = ['proxy_latitude', 'proxy_longitude'])
).set_index('datetime')

# Select weeks in spring 2015
stations_2015 = bikes.loc['2015-04-25'][
  ['proxy_latitude', 'proxy_longitude', 'geometry']
].drop_duplicates('geometry').reset_index(drop = True)
bikes_2015 = bikes.loc['2015-04-25':'2015-05-22'].reset_index().merge(
  stations_2015[['geometry']],
  on = 'geometry',
  how = 'right'
).set_index('datetime')

# Compute map boundaries
north = stations_2015['proxy_latitude'].sort_values().tail(1).iloc[0]
south = stations_2015['proxy_latitude'].sort_values().head(1).iloc[0]
east = stations_2015['proxy_longitude'].sort_values().tail(1).iloc[0]
west = stations_2015['proxy_longitude'].sort_values().head(1).iloc[0]

top = stations_2015[
  stations_2015['proxy_latitude'] == north
]['geometry'].iloc[0].y + 550
bottom = stations_2015[
  stations_2015['proxy_latitude'] == south
]['geometry'].iloc[0].y - 600
right = stations_2015[
  stations_2015['proxy_longitude'] == east
]['geometry'].iloc[0].x + 700
left = stations_2015[
  stations_2015['proxy_longitude'] == west
]['geometry'].iloc[0].x - 500

# Plot map
figure, (colorbar, background) = mpl.pyplot.subplots(
  nrows = 1,
  ncols = 2,
  gridspec_kw = {'width_ratios': [0.025, 0.975]},
  figsize = (2.5 * (right - left) / (top - bottom) / 0.975, 2.5)
)
background.set_ylim(bottom, top)
background.set_xlim(left, right)

colorbar = figure.add_axes((0,0,0.025,1), frameon = False)

for style in ['terrain-background', 'terrain-lines']:
  ctx.add_basemap(
    background,
    source = (
      'https://stamen-tiles-{s}.a.ssl.fastly.net/' +
      style +
      '/{z}/{x}/{y}{r}.png'
    ),
    zoom = 14,
    interpolation = None
  )

background.annotate(
  'Map tiles by Stamen Design (CC BY 3.0) -- '
  'Map data by © OpenStreetMap contributors (ODbL)',
  (0.995, 0.997),
  ha = 'right',
  va = 'top',
  xycoords = 'axes fraction',
  rotation = 'vertical',
  ma = 'right',
  color = '#E0E9F0',
  size = 2.25,
  clip_on = True
)

background.annotate(
  'Percentage of available\nbikes in Divvy stations',
  (0.945, 0.994),
  ha = 'right',
  va = 'top',
  xycoords = 'axes fraction',
  ma = 'right',
  color = '#F1F5F8',
  size = 2.7,
  linespacing = 1.5,
  clip_on = True
)

colorbar.annotate(
  '0%',
  (0.59, 0.004),
  ha = 'center',
  va = 'bottom',
  xycoords = 'axes fraction',
  rotation = 'vertical',
  ma = 'left',
  color = '#454545',
  weight = 'bold',
  size = 2.6,
  clip_on = True
)

colorbar.annotate(
  '100%',
  (0.59, 0.997),
  ha = 'center',
  va = 'top',
  xycoords = 'axes fraction',
  rotation = 'vertical',
  ma = 'right',
  color = '#C8C8C8',
  weight = 'bold',
  size = 2.6,
  clip_on = True
)

blank = bikes_2015.iloc[[0]]
blank = blank.assign(**{'geometry': blank['geometry'].translate(10**6,10**6)})

markers = figure.add_axes((0.025,0,0.975,1), frameon = False)
markers.set_ylim(bottom, top)
markers.set_xlim(left, right)
blank.plot(
  column = 'available_bikes_percent',
  vmin = 0,
  vmax = 100,
  ax = markers,
  markersize = 0.2,
  cmap = mpl.pyplot.get_cmap('winter_r', 2**6),
  aspect = None,
  legend = True,
  legend_kwds = dict(
    cax = colorbar,
    ticks = []
  )
)

figure.savefig(f'plots/chicago.png')

colorbar.remove()
background.remove()

# Select rows with bike availability changes
changes_2015 = bikes_2015.groupby(
  ['proxy_latitude', 'proxy_longitude']
).apply(
  lambda group: group.loc[
    group['available_bikes_percent'].shift() != group['available_bikes_percent']
  ]
).reset_index(
  level = ['proxy_latitude', 'proxy_longitude'],
  drop = True
)

# Plot stations for each datetime
for datetime in bikes_2015.index.unique():

  file_time = datetime.strftime('%Y_%m_%d_%H')

  markers.clear()
  markers.set_ylim(bottom, top)
  markers.set_xlim(left, right)

  markers.annotate(
    datetime.strftime('%I %p - %a %d %b %Y'),
    (0.945, 0.954),
    ha = 'right',
    va = 'top',
    xycoords = 'axes fraction',
    ma = 'right',
    color = '#F1F5F8',
    size = 2.7,
    linespacing = 1.5,
    clip_on = True
  )

  figure.savefig(f'plots/dates/date_{file_time}.png')

  markers.clear()
  markers.set_ylim(bottom, top)
  markers.set_xlim(left, right)

  if datetime in changes_2015.index:
    changes_2015.loc[[datetime]].plot(
      column = 'available_bikes_percent',
      vmin = 0,
      vmax = 100,
      ax = markers,
      markersize = 0.3,
      cmap = mpl.pyplot.get_cmap('winter_r', 2**6),
      aspect = None
    )

  figure.savefig(f'plots/stations/stations_{file_time}.png')

  markers.clear()

# Create a gif with imagemagick
subprocess.run(
  '''
  convert \
  +repage -fuzz 10% -quality 100 -delay 12 plots/chicago.png \
  null: plots/dates/*.png -layers composite \
  null: \( plots/stations/*.png -coalesce \) -dispose none -layers composite \
  -layers optimize-transparency \
  plots/stations_map.gif
  ''',
  shell = True
)
