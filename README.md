
<br>

### Links

 - [Code repository](https://www.github.com/chbk/time-series-forecasting)
 - [Online article](https://chbk.github.io/time-series-forecasting)

<br>

### Table of Contents

- [Bikes for thee, not for me](#bikes-for-thee-not-for-me)
- [The world of data science](#the-world-of-data-science)
- [Integrating data](#1-integrating-data)
- [Synthesizing data](#2-synthesizing-data)
- [Visualizing data](#3-visualizing-data)
- [Transforming data](#4-transforming-data)
- [Sampling data](#5-sampling-data)
- [Predicting data](#6-predicting-data)
- [Bikes for thee, and for me](#bikes-for-thee-and-for-me)

<br><br>

## Bikes for thee, not for me

Bikes are great. On sunny days, I love riding while enjoying a cool breeze. And with bike stations scattered all around the city, I can just stroll towards the closest one for some daily errands.

Or so I thought...

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/empty_station.jpg?raw=true"/></p>

The station is empty. I should have gotten there sooner. Or should I? Maybe if I wait a while, a bike will come to me? Could I predict when the bike station will run out of bikes?

Unfortunately, I've been confronted with this issue enough times that I started framing it as a data science problem (instead of buying my own bike). My first instinct was to look for some data to leverage for a divination session:

- [Archive of trips taken between bike stations](https://divvy-tripdata.s3.amazonaws.com/index.html)
- [Historical weather data](https://openweathermap.org)

If you too are frustrated at the prospect of getting stranded in an empty bike station, tag along, I've got a ride for you. A ride into...

<br>

## ...The world of data science

A powerful tool to extract actionable insights from data. In our case, using data archives of hourly weather records and times of trips between bike stations, we'll attempt to answer this question: "Given the weather, the location of the bike station, and the number of bikes in the station, how many bikes will be available in the future?"

Our journey will take us through these steps:

1. Integrating data
2. Synthesizing data
3. Visualizing data
4. Transforming data
5. Sampling data
6. Predicting data

In each section I'll mention what is being done and why it's being done. If you are interested in how things are done, check out the code in the source files.
Each file corresponds to a section, and can easily be read along. If you'd like to reproduce the results, you'll need data from the links above, a working installation of sqlite, and the following python packages:

    python3 -m pip install contextily geopandas haversine numba plotly pytorch seaborn tqdm

<br>

## 1. Integrating data

To bake a cake, you need processed ingredients. You wouldn't just toss raw wheat and cocoa beans in the oven and call it a day. You need to refine them first. And it's the same for data science. In this section, we'll refine and combine our raw data to make it palatable.

### Processing the weather

The weather file has one row per hour with various measurements. We parse it to select columns, fill missing values and format the datetime.

<div align="center">

<table>
  <thead>
    <tr>
      <th>datetime</th>
      <th>temperature (°C)</th>
      <th>humidity (%)</th>
      <th>cloudiness (%)</th>
      <th>wind speed (m/s)</th>
      <th>rain (mm/h)</th>
      <th>snow (mm/h)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>2015-05-10 08:00:00</td><td>8.3</td><td>89</td><td>90</td><td>4.1</td><td>0.3</td><td>0</td></tr>
    <tr><td>2015-05-10 09:00:00</td><td>8.25</td><td>92</td><td>90</td><td>3.1</td><td>0</td><td>0</td></tr>
    <tr><td>2015-05-10 10:00:00</td><td>8.01</td><td>92</td><td>90</td><td>4.1</td><td>0</td><td>0</td></tr>
    <tr><td>2015-05-10 11:00:00</td><td>7.75</td><td>96</td><td>90</td><td>4.1</td><td>0.3</td><td>0</td></tr>
    <tr><td>2015-05-10 12:00:00</td><td>7.78</td><td>96</td><td>90</td><td>5.1</td><td>2</td><td>0</td></tr>
  </tbody>
</table>

</div>

### Processing stations

Each file is a snapshot of different stations, with their coordinates, name, and maximum capacity.

<div align="center">

<table>
  <thead>
    <tr>
      <th>id</th>
      <th>name</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>capacity</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>119</td><td>Ashland Ave & Lake St</td><td>41.88541</td><td>-87.66732</td><td>19</td></tr>
    <tr><td>120</td><td>Wentworth Ave & Archer Ave</td><td>41.854564</td><td>-87.631937</td><td>15</td></tr>
    <tr><td>121</td><td>Blackstone Ave & Hyde Park Blvd</td><td>41.802562</td><td>-87.590368</td><td>15</td></tr>
    <tr><td>122</td><td>Ogden Ave & Congress Pkwy</td><td>41.87501</td><td>-87.67328</td><td>15</td></tr>
    <tr><td>123</td><td>California Ave & Milwaukee Ave</td><td>41.922695</td><td>-87.697153</td><td>15</td></tr>
  </tbody>
</table>

</div>

 Unfortunately, there are inconsistencies between snapshots. Some station properties change across time! Upon further investigation, we observe:

- Between two snapshot dates, a same station id can have different names.
- Between two snapshot dates, a same station id can have different capacities.
- Between two snapshot dates, a same station id can be over 90 meters apart.
- At any given snapshot date, all stations are more than 90 meters apart.
- At any given snapshot date, all stations have a unique station id.

To solve this, we assume that stations at most 90 meters apart with a same id and capacity are the same station, and all other stations are different stations. We assign proxy coordinates to each station, which are its average coordinates across all snapshot dates. This gives us a more consistent table.

<div align="center">

<table>
  <thead>
    <tr>
      <th>snapshot id</th>
      <th>station id</th>
      <th>name</th>
      <th>capacity</th>
      <th>proxy latitude</th>
      <th>proxy longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>887</td><td>119</td><td>Ashland Ave & Lake St</td><td>19</td><td>41.88541</td><td>-87.66732</td></tr>
    <tr><td>896</td><td>120</td><td>Wentworth Ave & Archer Ave</td><td>15</td><td>41.854564</td><td>-87.631937</td></tr>
    <tr><td>905</td><td>121</td><td>Blackstone Ave & Hyde Park Blvd</td><td>15</td><td>41.802562</td><td>-87.590368</td></tr>
    <tr><td>914</td><td>122</td><td>Ogden Ave & Congress Pkwy</td><td>15</td><td>41.87501</td><td>-87.67328</td></tr>
    <tr><td>923</td><td>123</td><td>California Ave & Milwaukee Ave</td><td>15</td><td>41.922695</td><td>-87.697153</td></tr>
  </tbody>
</table>

</div>

### Processing trips

Each file is a quarterly report of bike trips. A row is a record of source/destination stations and departure/arrival times.

<div align="center">

<table>
  <thead>
    <tr>
      <th>start datetime</th>
      <th>stop datetime</th>
      <th>start station id</th>
      <th>start station name</th>
      <th>stop station id</th>
      <th>stop station name</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>5/10/2015 9:25</td><td>5/10/2015 9:34</td><td>419</td><td>Lake Park Ave & 53rd St</td><td>423</td><td>University Ave & 57th St</td></tr>
    <tr><td>5/10/2015 9:24</td><td>5/10/2015 9:26</td><td>128</td><td>Damen Ave & Chicago Ave</td><td>214</td><td>Damen Ave & Grand Ave</td></tr>
    <tr><td>5/10/2015 9:27</td><td>5/10/2015 9:43</td><td>210</td><td>Ashland Ave & Division St</td><td>126</td><td>Clark St & North Ave</td></tr>
    <tr><td>5/10/2015 9:27</td><td>5/10/2015 9:35</td><td>13</td><td>Wilton Ave & Diversey Pkwy</td><td>327</td><td>Sheffield Ave & Webster Ave</td></tr>
    <tr><td>5/10/2015 9:28</td><td>5/10/2015 9:37</td><td>226</td><td>Racine Ave & Belmont Ave</td><td>223</td><td>Clifton Ave & Armitage Ave</td></tr>
  </tbody>
</table>

</div>

Since we want trips between two locations, we must merge them with the stations table. Unfortunately, station ids aren't reliable over time, so we use three properties to accurately join trips and stations: the station id, the station name, and the date of the trip. This operation is implemented in SQL which is more efficient for complex merging. The result is a list of trip dates and station snapshots.

<div align="center">

<table>
  <thead>
    <tr>
      <th>start datetime</th>
      <th>stop datetime</th>
      <th>start station snapshot id</th>
      <th>stop station snapshot id</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>2015-05-10 09:25:00</td><td>2015-05-10 09:34:00</td><td>3199</td><td>3223</td></tr>
    <tr><td>2015-05-10 09:24:00</td><td>2015-05-10 09:26:00</td><td>965</td><td>1673</td></tr>
    <tr><td>2015-05-10 09:27:00</td><td>2015-05-10 09:43:00</td><td>1637</td><td>947</td></tr>
    <tr><td>2015-05-10 09:27:00</td><td>2015-05-10 09:35:00</td><td>59</td><td>2651</td></tr>
    <tr><td>2015-05-10 09:28:00</td><td>2015-05-10 09:37:00</td><td>1772</td><td>1745</td></tr>
  </tbody>
</table>

</div>

<br>

## 2. Synthesizing data

We now have a coherent set of processed ingredients. But wait, something's missing! Our question was: "Given the weather, the location of the bike station, and the number of bikes in the station, how many bikes will be available in the future?"

We have the weather and locations of bike stations, but we don't have the number of bikes in each station. Is there a way to figure that out? Actually, we do have trips between stations, so we can infer the number of available bikes by summing arrivals and departures. We can also use each station's maximum capacity as a ceiling for the sum of incoming and outgoing bikes.

This operation requires a loop to incrementally sum and ceil the number of available bikes. Using numba, the loop is compiled to run at native machine code speed. Finally, the missing piece is revealed.

<div align="center">

<table>
  <thead>
    <tr>
      <th>proxy id</th>
      <th>datetime</th>
      <th>available bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>217</td><td>2015-05-10 09:00:00</td><td>12</td></tr>
    <tr><td>217</td><td>2015-05-10 10:00:00</td><td>11</td></tr>
    <tr><td>217</td><td>2015-05-10 11:00:00</td><td>12</td></tr>
    <tr><td>217</td><td>2015-05-10 12:00:00</td><td>13</td></tr>
    <tr><td>217</td><td>2015-05-10 13:00:00</td><td>15</td></tr>
  </tbody>
</table>

</div>

<br>

## 3. Visualizing data

Data are the pigments we use to paint a beautiful canvas conveying a story. So far, we've looked at our data in tabular format, with lonesome dimensions spread across columns. Let's blend these dimensions together to get a comprehensive view of bike availability across space and time.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/stations_map.gif?raw=true"/></p>

A good graph is self-explanatory, but for clarity's sake, here's a description of what we see. Each point is a bike station in Chicago. A point's color changes with the percentage of bikes available in the station, green when the station is empty, and blue when the station is full. What's remarkable is that we already clearly see a pattern! On weekdays, downtown gets filled with bikes, breathing in all the commuters in the morning, and sending them back home in the evening.

<br>

## 4. Transforming data

Remember our processed ingredients? This is the part where we whip the eggs and weigh the chocolate to gather primed and balanced amounts. Transforming and scaling are necessary to provide understandable and comparable data to our prediction model.

Let's start by scaling the percentages of available bikes to a range between 0 and 1.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/transformed_bikes.png?raw=true"/></p>

Next, we project station latitudes and longitudes to Cartesian coordinates and normalize them. We also compute the distance of each station to the city center and scale that.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/transformed_stations.png?raw=true"/></p>

What is 3 - 23? If you say -20 you are technically correct, but semantically wrong. When comparing times, 3 and 23 are just 4 hours apart! Unfortunately, a machine only understands technicalities. So how do we connect times at midnight? We transform hours into cyclical features that continuously oscillate between -1 and 1. To tell all the hours apart, we use both sine and cosine transformations, so every hour has a unique pair of mappings.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/transformed_time.png?raw=true"/></p>

We also scale temperature, humidity, cloudiness, wind speed, rain, and snow, so they all lie approximately in the same range between 0 and 1.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/transformed_weather.png?raw=true"/></p>

From those distributions we notice that many rain and snow measurements equal 0. Our dataset seems skewed. When training our model, we must make sure to give it enough cases of rainy and snowy days so it learns to pay attention to those features.

<br>

## 5. Sampling data

 One way to obtain a homogenous dataset is to sample it to select enough measurements in each category. First let's see how our data is distributed. We segment it into overlapping 24 hour sequences that will be used to train our model.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/unbalanced_seasons.png?raw=true"/></p>

We have different numbers of sequences in each season and there are less wet days than dry days. However, the differences aren't extreme. We still have lots of data in each category, which should be enough to train our model. Notice that while there are very few actual hours with rain or snow, there are considerably more wet sequences with some amount of precipitation. Viewing data is a matter of perspective.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/unbalanced_variation.png?raw=true"/></p>

Bike availability is the target feature we want to predict. Unfortunately, many bike stations keep the same number of bikes over long periods of time. That means there's a risk our model will predict an unvarying number of bikes, whatever the input sequence! To avoid this, we must select a uniform subset of data including enough cases of stations that are often visited throughout the day.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/sampled_variation.png?raw=true"/></p>

After sampling at random within each interval, we get a much more uniform dataset. Having plenty of different cases will help the model learn better.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/sampled_seasons.png?raw=true"/></p>

Going deeper, we can cross seasons and variation to determine how they are correlated. Unsurprisingly, we observe less variation in winter when it's cold. And even less on wet days when it's snowing. After all, there aren't many bike stations getting visited in a blizzard in the middle of December. Can our model  understand that to make more accurate predictions?

<br>

## 6. Predicting data

Now that all our ingredients are ready, we can move on to the next step: baking our model. What kind of model do we want? One that can predict time series of bike availability from other sequences. Let's find a mold for that.

### Choosing the model architecture

For a neural network, there are several different architectures suitable for sequence prediction. Our main contenders are:

- The recurrent neural network (RNN)
- The long short-term memory network (LSTM)
- The transformer

Here's a quick and basic overview of each architecture.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/model_rnn.png?raw=true"/></p>

A RNN is a model that updates its state recursively at each sequence step, by taking into account its previous state and the current sequence item. Recurrent neural networks suffer from long-term memory loss because they continuously focus on the last provided inputs, which eventually override longer dependencies. Basically, you can see a RNN as a goldfish that constantly rediscovers its environment.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/model_lstm.png?raw=true"/></p>

A solution to long-term memory loss is the LSTM, which is a RNN upgraded with two internal states: a short-term state that grasps the most recent developments, and a long-term state that can ignore sequence items as they are encountered. It's still a goldfish, but now it keeps a diary. This helps preserve long-term dependencies, but stays computationally expensive as the sequence must be processed recursively.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/model_transformer.png?raw=true"/></p>

Enters the transformer, a state-of-the-art architecture for sequence processing and prediction. It solves both long-term memory loss and computational cost by eliminating recursion during training. Each sequence item is appended to a positional encoding that enables  processing the entire sequence as a whole. This is all done in parallel with multiple attention heads, each one trained to understand a specific relationship between sequence items. Like a school of cognitively-enhanced, higher-dimensional goldfish, processing all time-bound events at once.

In the original paper introducing the transformer [[1](https://dl.acm.org/doi/10.5555/3295222.3295349)], the architecture is used for machine translation between languages. In other words, it maps one sequence to another sequence. This seems promising for our use case, except that we don't have just one, but multiple sequences for bike availability and weather, in addition to some constants like station coordinates.

Looking for available implementations of the transformer architecture tweaked to work with multivariate data didn't yield anything directly usable, but there is a paper that focuses on solving that for spatio-temporal forecasting [[2](https://arxiv.org/abs/2109.12218)]. They concatenate sequences together, along with variable embeddings, before inputting them to the network. They also introduce two new modules: global attention to examine relationships between all sequences mutually, and local attention to find relationships within each sequence individually. This seems to suit our needs, so let's apply these concepts to build a fully functional model.

### Implementing the model

Taking inspiration from the above sources and tuning them to simplify and generalize, the end result is an architecture similar to a vanilla transformer, but modified to accommodate multiple variables with the following enhancements:

- An additional dimension to separate variables in attention modules.
- Different modules for local and global attention among variables.

This way, we can avoid concatenating sequences in one big matrix along with variable embeddings. Every variable is processed individually, except when they are pitched against each other in the global attention modules.

To explain further, here is how an attention head works. First, the sequence is projected into query (Q), key (K), and value (V) matrices. Each row corresponds to a sequence item. Query and key matrices are then multiplied to yield a score (S) matrix, where each row is softmaxed to emphasize the strongest relationships between sequence items. The score matrix is then multiplied with the value matrix to yield an encoding (E) of the original sequence.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/attention_original.png?raw=true"/></p>

In our case, we have multiple variables, so each attention head gets an additional dimension to accommodate the sequences. The operations are the same as in the original transformer, but parallelized for multiple variables (A, B, C). This is how an attention head behaves in our local attention module.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/attention_local.png?raw=true"/></p>

Local attention modules learn relationships within each sequence, but the model must also figure out relationships between sequences. To do so, attention heads can concatenate key projections and value projections to reduce their number of dimensions. This effectively allows each sequence item to encounter every other sequence item of every variable. The strengths of their relationships are directly available in the score matrices. This is how an attention head behaves in our global attention module.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/attention_global.png?raw=true"/></p>

### Training the model

The architecture is implemented but the model is completely ignorant at this point. It hasn't seen any data, it doesn't know anything about the world. The next step is to train it with a part of the dataset that we call the training-and-validation set. The training-and-validation set is like a collection of corrected exam papers with questions and answers. In our case, the "questions" are the number of bikes, weather, and location, and the "answers" are the future number of bikes. The goal of training is to teach the model to get the correct answers to questions. Like a student studying diligently by practicing on past exams every day, the entire training-and-validation set is provided to the model at every epoch:

1. First, the collection of exam questions are split into a large training set and small validation set.
2. The model reads some questions from the training set and attempts to answer them.
3. The model mentally adjusts its knowledge depending on how well it answered.
4. Repeat steps 2 and 3 until the entire training set has been seen.
5. The model takes a mock exam on the validation set and attempts to answer without peeking.
6. The model's answers on the mock exam provide an estimate of its performance.

Training our model on a subset of data to predict the next 6 hours from the previous 18 hours shows that it learns and stabilizes rather quickly. Even though the training-and-validation set is shuffled at each epoch, the validation error stays above the training error. That's because recursion is eliminated during training, so for each hour to predict, the model sees the entire sequence that comes before, but during validation, the model only sees the input sequence and has to predict all the following hours on its own.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/prediction_error.png?raw=true"/></p>

After the model has trained enough times, it is deemed ready to take the final exam on the testing set. The testing set is the other part of the dataset that the model has not yet encountered. This exam really measures the intelligence of the model. If it learned the answers of past exams by heart, it will fail on new unseen questions. But if it correctly understood relationships between questions and answers, it will be competent enough to generalize to novel cases. The model passes the test with an average error of around 4%, or about 1 bike of error in a station of 25 bikes. Not bad at all! Check out some of its predictions below.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/prediction_01.png?raw=true"/></p>

Prediction for a station on a workday in autumn. The model understands this station is highly visited in the morning, and predicts a trend accordingly.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/prediction_02.png?raw=true"/></p>

Prediction for a station during a winter weekend. The model successfully guesses a slight upward trend throughout the day.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/prediction_03.png?raw=true"/></p>

Prediction for a workday in spring. The prognostic is a downward trend during the afternoon, which comes true.

<p align="center"><img src="https://raw.githubusercontent.com/chbk/time-series-forecasting/images/prediction_04.png?raw=true"/></p>

Another prediction for a workday in spring, with some amount of rain. The model accurately predicts little variation for this station.

<br>

## Bikes for thee, and for me

This concludes our journey through the realms of data science. Transformers work really well for sequence generation. If you're interested in reading more about that, check out the references below.

<br>

1. Vaswani et al. (2017) [Attention Is All You Need](https://dl.acm.org/doi/10.5555/3295222.3295349)
2. Grigsby et al. (2021) [Long-Range Transformers for Dynamic Spatiotemporal Forecasting](https://arxiv.org/abs/2109.12218)
