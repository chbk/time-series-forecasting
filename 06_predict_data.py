#!/usr/bin/env python3

import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib as mpl
from typing import Optional, Callable, Tuple

#%% Model ----------------------------------------------------------------------

class MultivariateLinear(torch.nn.Module):

  def __init__(
    self,
    variable_count: int,
    input_width: int,
    output_width: int
  ) -> None:
    super().__init__()
    self.weights = torch.nn.Parameter(torch.empty(
      (variable_count, input_width, output_width)
    ))
    self.biases = torch.nn.Parameter(torch.empty(
      (variable_count, 1, output_width)
    ))
    if variable_count > 0:
      torch.nn.init.kaiming_uniform_(self.weights, a = 5 ** 0.5)
      fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
      bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
      torch.nn.init.uniform_(self.biases, -bound, bound)

  def forward(
    self,
    # [ batches, variables, input length, input width ]
    input: torch.Tensor
  ) -> torch.Tensor:
    return input @ self.weights[:input.shape[-3]] + self.biases[:input.shape[-3]]

class MultivariateAttention(torch.nn.Module):

  def __init__(
    self,
    embedding_width: int = 16,
    attention_width: int = 16,
    sequence_count: int = 1,
    constant_count: int = 0,
    head_count: int = 8,
    # Pay attention to variables independently (True) or mutually (False)
    local: bool = False
  ) -> None:

    super().__init__()

    self.local = local

    dimensions = (sequence_count, head_count, embedding_width, attention_width)

    self.query_sequence_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.query_sequence_weights)

    self.key_sequence_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.key_sequence_weights)

    self.value_sequence_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.value_sequence_weights)

    dimensions = (constant_count, head_count, embedding_width, attention_width)

    self.query_constant_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.query_constant_weights)

    self.key_constant_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.key_constant_weights)

    self.value_constant_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.value_constant_weights)

    dimensions = (sequence_count, head_count * attention_width, embedding_width)

    self.sequence_head_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.sequence_head_weights)

    dimensions = (constant_count, head_count * attention_width, embedding_width)

    self.constant_head_weights = torch.nn.Parameter(torch.empty(dimensions))
    torch.nn.init.xavier_uniform_(self.constant_head_weights)

  def forward(
    self,
    # [ batches, sequences, sequence length, embedding width ]
    query_sequences: torch.Tensor,
    # [ batches, sequences, sequence length, embedding width ]
    key_sequences: Optional[torch.Tensor] = None,
    # [ batches, sequences, sequence length, embedding width ]
    value_sequences: Optional[torch.Tensor] = None,
    # [ batches, constants, 1, embedding width ]
    query_constants: Optional[torch.Tensor] = None,
    # [ batches, constants, 1, embedding width ]
    key_constants: Optional[torch.Tensor] = None,
    # [ batches, constants, 1, embedding width ]
    value_constants: Optional[torch.Tensor] = None,
    # Mask the future from the past in sequences
    mask: bool = False
  ) -> Tuple[torch.Tensor, torch.Tensor]:

    if query_constants is None:
      query_constants = torch.zeros(*query_sequences.shape[:-3], 0, 1, query_sequences.shape[-1])

    if self.local:
      if key_sequences is None: key_sequences = torch.zeros_like(query_sequences)
      if key_constants is None: key_constants = torch.zeros_like(query_constants)
      if value_sequences is None: value_sequences = torch.zeros_like(query_sequences)
      if value_constants is None: value_constants = torch.zeros_like(query_constants)

    if key_sequences is None:
      key_sequences = torch.zeros(*query_sequences.shape[:-3], 0, *query_sequences.shape[-2:])
    if key_constants is None:
      key_constants = torch.zeros(*query_constants.shape[:-3], 0, 1, query_constants.shape[-1])
    if value_sequences is None:
      value_sequences = torch.zeros(*query_sequences.shape[:-3], 0, *query_sequences.shape[-2:])
    if value_constants is None:
      value_constants = torch.zeros(*query_constants.shape[:-3], 0, 1, query_constants.shape[-1])

    query_sequence_length = query_sequences.shape[-2]
    query_sequence_count = query_sequences.shape[-3]
    query_constant_count = query_constants.shape[-3]

    if self.local:
      query_constants = 0 * query_constants
      key_sequences = key_sequences[...,:query_sequence_count,:,:]
      key_constants = key_constants[...,:query_constant_count,:,:]
      value_sequences = value_sequences[...,:query_sequence_count,:,:]
      value_constants = value_constants[...,:query_constant_count,:,:]

    key_sequence_length = key_sequences.shape[-2]
    key_sequence_count = key_sequences.shape[-3]
    key_constant_count = key_constants.shape[-3]

    # Insert extra dimension for attention heads
    # [ batches, variables, heads, variable length, embedding width ]
    query_sequences = query_sequences.unsqueeze(-3)
    query_constants = query_constants.unsqueeze(-3)
    key_sequences = key_sequences.unsqueeze(-3)
    key_constants = key_constants.unsqueeze(-3)
    value_sequences = value_sequences.unsqueeze(-3)
    value_constants = value_constants.unsqueeze(-3)

    # Encode
    query_sequence_encodings = query_sequences @ self.query_sequence_weights[:query_sequence_count]
    query_constant_encodings = query_constants @ self.query_constant_weights[:query_constant_count]
    key_sequence_encodings = key_sequences @ self.key_sequence_weights[:key_sequence_count]
    key_constant_encodings = key_constants @ self.key_constant_weights[:key_constant_count]
    value_sequence_encodings = value_sequences @ self.value_sequence_weights[:key_sequence_count]
    value_constant_encodings = value_constants @ self.value_constant_weights[:key_constant_count]

    # Concatenate variables
    if not self.local:

      key_sequence_encodings = key_constant_encodings = torch.cat(
        [
          key_sequence_encodings.transpose(-4, -3).flatten(-3, -2),
          key_constant_encodings.transpose(-4, -3).flatten(-3, -2)
        ],
        dim = -2
      ).unsqueeze(-4)
      value_sequence_encodings = value_constant_encodings = torch.cat(
        [
          value_sequence_encodings.transpose(-4, -3).flatten(-3, -2),
          value_constant_encodings.transpose(-4, -3).flatten(-3, -2)
        ],
        dim = -2
      ).unsqueeze(-4)

    # Compute attention
    sequences = query_sequence_encodings @ key_sequence_encodings.transpose(-2, -1)
    constants = query_constant_encodings @ key_constant_encodings.transpose(-2, -1)

    sequences = sequences / (query_sequence_encodings.shape[-1] ** 0.5)
    constants = constants / (query_constant_encodings.shape[-1] ** 0.5)

    # Hide the future from the past
    if mask:
      mask = (-torch.inf * torch.ones(query_sequence_length, key_sequence_length)).triu(1)
      if not self.local:
        mask = mask.repeat(1, key_sequence_count)
        mask = torch.cat([mask, torch.zeros(query_sequence_length, key_constant_count)], -1)
      sequences = sequences + mask

    # Compute scores
    sequences = torch.softmax(sequences, -1)
    constants = torch.softmax(constants, -1) if not self.local else torch.Tensor([[1]])

    # Compute outputs
    sequences = sequences @ value_sequence_encodings
    constants = constants @ value_constant_encodings

    # Concatenate heads
    sequences = sequences.transpose(-2, -3).flatten(-2)
    constants = constants.transpose(-2, -3).flatten(-2)

    # Coalesce heads
    sequences = sequences @ self.sequence_head_weights[:query_sequence_count]
    constants = constants @ self.constant_head_weights[:query_constant_count]

    return sequences, constants

class MultivariateTransformerLayer(torch.nn.Module):

  def __init__(
    self,
    embedding_width: int = 16,
    attention_width: int = 16,
    feedforward_width: int = 256,
    sequence_count: int = 1,
    constant_count: int = 0,
    head_count: int = 8,
    dropout: float = 0.1,
  ) -> None:

    super().__init__()

    # Local attention
    self.local_attention = MultivariateAttention(
      embedding_width = embedding_width,
      attention_width = attention_width,
      sequence_count = sequence_count,
      constant_count = constant_count,
      head_count = head_count,
      local = True
    )
    self.local_dropout = torch.nn.Dropout(dropout)
    self.local_sequences_layernorm = torch.nn.LayerNorm(embedding_width)
    self.local_constants_layernorm = torch.nn.LayerNorm(embedding_width)

    # Global attention
    self.global_attention = MultivariateAttention(
      embedding_width = embedding_width,
      attention_width = attention_width,
      sequence_count = sequence_count,
      constant_count = constant_count,
      head_count = head_count
    )
    self.global_dropout = torch.nn.Dropout(dropout)
    self.global_sequences_layernorm = torch.nn.LayerNorm(embedding_width)
    self.global_constants_layernorm = torch.nn.LayerNorm(embedding_width)

    # Local cross attention
    self.local_cross_attention = MultivariateAttention(
      embedding_width = embedding_width,
      attention_width = attention_width,
      sequence_count = sequence_count,
      constant_count = constant_count,
      head_count = head_count,
      local = True
    )
    self.local_cross_dropout = torch.nn.Dropout(dropout)
    self.local_sequences_cross_layernorm = torch.nn.LayerNorm(embedding_width)
    self.local_constants_cross_layernorm = torch.nn.LayerNorm(embedding_width)

    # Global cross attention
    self.global_cross_attention = MultivariateAttention(
      embedding_width = embedding_width,
      attention_width = attention_width,
      sequence_count = sequence_count,
      constant_count = constant_count,
      head_count = head_count
    )
    self.global_cross_dropout = torch.nn.Dropout(dropout)
    self.global_cross_sequences_layernorm = torch.nn.LayerNorm(embedding_width)
    self.global_cross_constants_layernorm = torch.nn.LayerNorm(embedding_width)

    # Sequences feedforward
    self.sequences_feedforward = torch.nn.Sequential(
      MultivariateLinear(sequence_count, embedding_width, feedforward_width),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      MultivariateLinear(sequence_count, feedforward_width, embedding_width)
    )
    self.sequences_feedforward_layernorm = torch.nn.LayerNorm(embedding_width)

    # Constants feedforward
    self.constants_feedforward = torch.nn.Sequential(
      MultivariateLinear(constant_count, embedding_width, feedforward_width),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      MultivariateLinear(constant_count, feedforward_width, embedding_width)
    )
    self.constants_feedforward_layernorm = torch.nn.LayerNorm(embedding_width)

  def forward(
    self,
    input_sequences: torch.Tensor,
    input_constants: Optional[torch.Tensor] = None,
    memory_sequences: Optional[torch.Tensor] = None,
    memory_constants: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:

    sequences = input_sequences
    constants = input_constants

    # Local attention
    s, c = self.local_attention(
      query_sequences = sequences,
      key_sequences = sequences,
      value_sequences = sequences,
      query_constants = constants,
      key_constants = constants,
      value_constants = constants,
      mask = True
    )

    sequences = sequences + self.local_dropout(s)
    sequences = self.local_sequences_layernorm(sequences)
    if constants is not None:
      constants = constants + self.local_dropout(c)
      constants = self.local_constants_layernorm(constants)

    # Global attention during encoding
    if (
      memory_sequences is None and
      memory_constants is None and
      (sequences.shape[-3] > 1 or constants is not None)
    ):

      s, c = self.global_attention(
        query_sequences = sequences,
        key_sequences = sequences,
        value_sequences = sequences,
        query_constants = constants,
        key_constants = constants,
        value_constants = constants,
        mask = True
      )

      sequences = sequences + self.global_dropout(s)
      sequences = self.global_sequences_layernorm(sequences)
      if constants is not None:
        constants = constants + self.global_dropout(c)
        constants = self.global_constants_layernorm(constants)

    # Local cross attention during decoding
    if memory_sequences is not None or memory_constants is not None:

      s, c = self.local_cross_attention(
        query_sequences = sequences,
        key_sequences = memory_sequences,
        value_sequences = memory_sequences,
        query_constants = constants,
        key_constants = memory_constants,
        value_constants = memory_constants,
        mask = False
      )

      sequences = sequences + self.local_cross_dropout(s)
      sequences = self.local_sequences_cross_layernorm(sequences)
      if constants is not None:
        constants = constants + self.local_cross_dropout(c)
        constants = self.local_constants_cross_layernorm(constants)

    # Global cross attention during decoding
    if (
      (memory_sequences is not None and memory_sequences.shape[-3] > 1) or
      memory_constants is not None
    ):

      s, c = self.global_cross_attention(
        query_sequences = sequences,
        key_sequences = memory_sequences,
        value_sequences = memory_sequences,
        query_constants = constants,
        key_constants = memory_constants,
        value_constants = memory_constants,
        mask = False
      )

      sequences = sequences + self.global_cross_dropout(s)
      sequences = self.global_cross_sequences_layernorm(sequences)
      if constants is not None:
        constants = constants + self.global_cross_dropout(c)
        constants = self.global_cross_constants_layernorm(constants)

    # Feedforward
    sequences = sequences + self.sequences_feedforward(sequences)
    sequences = self.sequences_feedforward_layernorm(sequences)
    if constants is not None:
      constants = constants + self.constants_feedforward(constants)
      constants = self.constants_feedforward_layernorm(constants)

    return sequences, constants

class MultivariateTransformer(torch.nn.Module):

  def __init__(
    self,
    sequence_count: int,
    sequence_width: int,
    constant_count: int = 0,
    constant_width: int = 0,
    sequence_length: int = 24,
    embedding_width: int = 16,
    attention_width: int = 16,
    feedforward_width: int = 256,
    head_count: int = 8,
    layer_count: int = 6,
    dropout: float = 0.1
  ) -> None:

    super().__init__()

    self.positions = torch.cat(
      (
        torch.sin(
          torch.arange(sequence_length) * 2 * np.pi / sequence_length - np.pi
        ).unsqueeze(1),
        torch.cos(
          torch.arange(sequence_length) * 2 * np.pi / sequence_length - np.pi
        ).unsqueeze(1)
      ),
      dim = -1
    )
    self.sequence_length = sequence_length

    self.sequences_embedding = MultivariateLinear(
      sequence_count, sequence_width + 2, embedding_width
    )

    self.constants_embedding = torch.nn.Sequential(
      MultivariateLinear(constant_count, constant_width, feedforward_width),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      MultivariateLinear(constant_count, feedforward_width, embedding_width)
    )

    self.encoder_layers = torch.nn.ModuleList(
      MultivariateTransformerLayer(
        embedding_width = embedding_width,
        attention_width = attention_width,
        feedforward_width = feedforward_width,
        sequence_count = sequence_count,
        constant_count = constant_count,
        head_count = head_count,
        dropout = dropout
      ) for _ in range(layer_count)
    )

    self.decoder_layers = torch.nn.ModuleList(
      MultivariateTransformerLayer(
        embedding_width = embedding_width,
        attention_width = attention_width,
        feedforward_width = feedforward_width,
        sequence_count = sequence_count,
        constant_count = constant_count,
        head_count = head_count,
        dropout = dropout
      ) for _ in range(layer_count)
    )

    self.output_projection = torch.nn.Sequential(
      MultivariateLinear(sequence_count, embedding_width, sequence_width),
      torch.nn.Tanh()
    )

  def forward(
    self,
    # [ batches, sequences, sequence length, sequence width ]
    source_sequences: torch.Tensor,
    # [ batches, sequences, sequence length, sequence width ]
    target_sequences: Optional[torch.Tensor] = None,
    # [ batches, constants, 1, constant width ]
    source_constants: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:

    # Get lengths and set empty target sequences
    source_sequence_length = source_sequences.shape[-2]
    if target_sequences is None:
      target_sequence_length = self.sequence_length - source_sequence_length
      target_sequences = torch.zeros(
        *source_sequences.shape[:-2], 0, source_sequences.shape[-1]
      )
    else:
      target_sequence_length = target_sequences.shape[-2]

    # Repeat positions for each source sequence
    source_positions = self.positions[
      :source_sequence_length
    ].repeat(*source_sequences.shape[:-2], 1, 1)

    # Repeat positions for each target sequence
    target_positions = self.positions[
      source_sequence_length:source_sequence_length + target_sequence_length
    ].repeat(*target_sequences.shape[:-2], 1, 1)

    # Append positions to source sequences
    source_sequences = torch.cat((source_sequences, source_positions), -1)

    # Select last entry of sequences as trigger sequences for the decoder
    source_sequences, trigger_sequences = source_sequences.split(
      [source_sequence_length - 1, 1],
      dim = -2
    )

    # Select trigger sequences for target sequences
    trigger_sequences = trigger_sequences[...,:target_sequences.shape[-3],:,:]

    # Embed sequences and constants
    x = self.sequences_embedding(source_sequences)
    c = source_constants
    if c is not None: c = self.constants_embedding(c)

    # Encode
    for layer in self.encoder_layers:
      x, c = layer(
        input_sequences = x,
        input_constants = c
      )

    # Decode
    y = target_sequences[...,:-1,:]
    start = target_sequence_length - 1 if self.training else 0
    for current_sequence_length in range(start, target_sequence_length):

      # Append positions to target sequences
      y = torch.cat((y, target_positions[...,:current_sequence_length,:]), -1)

      # Prepend trigger sequences to target sequences
      y = torch.cat((trigger_sequences, y), -2)

      # Embed sequences
      y = self.sequences_embedding(y)

      # Multi-head attention
      for layer in self.decoder_layers:
        y, _ = layer(
          input_sequences = y,
          memory_sequences = x,
          memory_constants = c
        )

      # Projection layer to output plain sequences from attention outputs
      y = self.output_projection(y) + 0.5

    return y

#%% Dataset --------------------------------------------------------------------

class BikesWeatherDataset(torch.utils.data.Dataset):

  def __init__(
    self,
    input: str,
    # Number of past hours to predict from
    source_length: int = 18,
    # Number of hours to predict
    target_length: int = 6,
    before_date: Optional[dt.datetime] = None,
    after_date: Optional[dt.datetime] = None
  ) -> None:

    super().__init__()

    data = pd.read_csv(
      input,
      parse_dates = ['datetime'],
      infer_datetime_format = True
    )

    # Select dates for training or testing set
    if before_date is not None:
      data = data[data['datetime'] < before_date]
    if after_date is not None:
      data = data[data['datetime'] > after_date]

    # Sample data for faster training on similar cases
    # data = data[data['downtown_distance'] <= 0.02].reset_index(drop = True)

    # Each row is an item in a sequence
    # self.indices[item index] = row index of the beginning of the sequence
    start_data = data[data['contiguous_length'] >= source_length + target_length]

    if before_date is not None:
      before_date -= dt.timedelta(hours = source_length + target_length - 1)
      start_data = start_data[start_data['datetime'] < before_date]

    self.indices = start_data.index.to_numpy()

    # Features
    features = [
      'available_bikes', # 0
      'x_coordinate', # 1
      'y_coordinate', # 2
      'z_coordinate', # 3
      'downtown_distance', # 4
      'sin_year_day', # 5
      'cos_year_day', # 6
      'sin_week_hour', # 7
      'cos_week_hour', # 8
      'sin_day_hour', # 9
      'cos_day_hour', # 10
      'workday', # 11
      'temperature', # 12
      'rain_1h', # 13
      'snow_1h', # 14
      'humidity', # 15
      'cloudiness', # 16
      'wind_speed', # 17
    ]

    data = data[features].to_numpy(dtype = 'float32')
    self.data = torch.from_numpy(data)

    self.source_length = source_length
    self.target_length = target_length

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(
    self,
    index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    index = self.indices[index]
    source_start = index
    source_stop = index + self.source_length
    target_start = source_stop
    target_stop = index + self.source_length + self.target_length

    source_sequences = torch.stack((
      # Available bikes
      torch.nn.functional.pad(self.data[source_start:source_stop,0:1], (0, 1)),
      # Day of year
      torch.nn.functional.pad(self.data[source_start:source_stop,5:7], (0, 0)),
      # Hour of week
      torch.nn.functional.pad(self.data[source_start:source_stop,7:9], (0, 0)),
      # Hour of day
      torch.nn.functional.pad(self.data[source_start:source_stop,9:11], (0, 0)),
      # Workday
      torch.nn.functional.pad(self.data[source_start:source_stop,11:12], (0, 1)),
      # Temperature
      torch.nn.functional.pad(self.data[source_start:source_stop,12:13], (0, 1)),
      # Rain
      torch.nn.functional.pad(self.data[source_start:source_stop,13:14], (0, 1)),
      # Snow
      torch.nn.functional.pad(self.data[source_start:source_stop,14:15], (0, 1)),
      # Humidity
      torch.nn.functional.pad(self.data[source_start:source_stop,15:16], (0, 1)),
      # Cloudiness
      torch.nn.functional.pad(self.data[source_start:source_stop,16:17], (0, 1)),
      # Wind speed
      torch.nn.functional.pad(self.data[source_start:source_stop,17:18], (0, 1))
    ))

    source_constants = torch.stack((
      # Location
      torch.nn.functional.pad(self.data[source_stop,1:4], (0, 0)),
      # Distance to downtown
      torch.nn.functional.pad(self.data[source_stop,4:5], (0, 2))
    )).unsqueeze(1)

    target_sequences = torch.stack((
      # Available bikes
      torch.nn.functional.pad(self.data[target_start:target_stop,0:1], (0, 1)),
    ))

    return source_sequences, source_constants, target_sequences

#%% Sample, train, test --------------------------------------------------------

def train_test_sample(
  training_dataset: torch.utils.data.Dataset,
  testing_dataset: torch.utils.data.Dataset,
  subset_size: int,
  split: float = 0.8,
  batch_size: int = 1,
  fold_count: int = 1
):

  training_size = min(int(split * subset_size), len(training_dataset))
  training_size = training_size - training_size % (batch_size * fold_count)
  testing_size = min(int((1 - split) * subset_size), len(testing_dataset))
  testing_size = testing_size - testing_size % (batch_size * fold_count)

  training_dataset = torch.utils.data.Subset(
    training_dataset,
    np.random.choice(range(len(training_dataset)), training_size, replace = False)
  )

  testing_dataset = torch.utils.data.Subset(
    testing_dataset,
    np.random.choice(range(len(testing_dataset)), testing_size, replace = False)
  )

  return training_dataset, testing_dataset

def train(
  dataset: torch.utils.data.Dataset,
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  loss: Callable,
  split: float = 0.8,
  batch_size: int = 1,
  fold_count: int = 1,
  epoch_count: int = 10
):

  if fold_count > 1: fold_size = int(len(dataset) / fold_count)
  else: fold_size = int(split * len(dataset))

  folds = [fold_size for _ in range(fold_count)]
  folds += [len(dataset) - sum(folds)]

  training_error = [[] for e in range(epoch_count)]
  validation_error = [[] for e in range(epoch_count)]

  for e in range(epoch_count):

    # Randomly select folds
    subsets = torch.utils.data.random_split(dataset, folds)

    # Iterate k times or once if dataset is not folded
    for k in range(*((fold_count,) if fold_count > 1 else (1, 2))):

      # Train
      model.train(True)
      for f in range(fold_count):

        # Skip if this fold is the validation fold
        if f == k: continue

        # Prepare subset
        loader = torch.utils.data.DataLoader(
          subsets[f],
          shuffle = True,
          batch_size = batch_size
        )

        # For each batch in this fold
        for b, batch in tqdm.tqdm(enumerate(loader)):

          source_sequences, source_constants, target_sequences = batch

          output_sequences = model(
            source_sequences = source_sequences,
            target_sequences = target_sequences,
            source_constants = source_constants
          )

          output_sequences = output_sequences[...,:target_sequences.shape[-3],:,:]

          error = loss(output_sequences, target_sequences)

          training_error[e] += [error.item()]

          # Backpropagate
          optimizer.zero_grad()
          error.backward()
          optimizer.step()

      # Validate
      model.train(False)
      with torch.no_grad():

        # Prepare subset
        loader = torch.utils.data.DataLoader(
          subsets[k],
          shuffle = True,
          batch_size = batch_size
        )

        # For each batch in this fold
        for b, batch in tqdm.tqdm(enumerate(loader)):

          source_sequences, source_constants, target_sequences = batch

          output_sequences = model(
            source_sequences = source_sequences,
            source_constants = source_constants
          )

          output_sequences = output_sequences[...,:target_sequences.shape[-3],:,:]

          error = loss(output_sequences, target_sequences)

          validation_error[e] += [error.item()]

    training_error[e] = np.mean(training_error[e])
    validation_error[e] = np.mean(validation_error[e])

    print(f'\nepochs: {e + 1}')
    print(f'training error: {np.round(training_error[e], 6)}')
    print(f'validation error: {np.round(validation_error[e], 6)}')

  return training_error, validation_error

def test(
  dataset: torch.utils.data.Dataset,
  model: torch.nn.Module,
  loss: Callable,
  batch_size: int = 1
):

  testing_error = []
  all_source_sequences = []
  all_source_constants = []
  all_target_sequences = []
  all_output_sequences = []

  # Test
  model.train(False)
  with torch.no_grad():

    loader = torch.utils.data.DataLoader(
      dataset,
      shuffle = False,
      batch_size = batch_size
    )

    for b, batch in tqdm.tqdm(enumerate(loader)):

      source_sequences, source_constants, target_sequences = batch

      output_sequences = model(
        source_sequences = source_sequences,
        source_constants = source_constants
      )

      output_sequences = output_sequences[...,:target_sequences.shape[-3],:,:]

      error = loss(output_sequences, target_sequences)

      testing_error += [error.item()]
      all_source_sequences += [source_sequences]
      all_source_constants += [source_constants]
      all_target_sequences += [target_sequences]
      all_output_sequences += [output_sequences]

    testing_error = np.mean(testing_error)
    source_sequences = torch.cat(all_source_sequences)
    source_constants = torch.cat(all_source_constants)
    target_sequences = torch.cat(all_target_sequences)
    output_sequences = torch.cat(all_output_sequences)

  return (
    source_sequences,
    source_constants,
    target_sequences,
    output_sequences,
    testing_error
  )

#%% Load data ------------------------------------------------------------------

training_dataset = BikesWeatherDataset(
  'data/divvy_bikes_chicago_weather_2013-2017_sampled.csv',
  before_date = dt.datetime(2016, 11, 15)
)

testing_dataset = BikesWeatherDataset(
  'data/divvy_bikes_chicago_weather_2013-2017_sampled.csv',
  after_date = dt.datetime(2016, 11, 15)
)

#%% Sample dataset --------------------------------------------------------------

training_dataset, testing_dataset = train_test_sample(
  training_dataset = training_dataset,
  testing_dataset = testing_dataset,
  subset_size = 80000, # Lower the subset size for shorter training
  batch_size = 16
)

#%% Check dataset --------------------------------------------------------------

i = 40

sin_hour = training_dataset[i][0][3,:,0].numpy()
cos_hour = training_dataset[i][0][3,:,1].numpy()
hours = np.round(24 * (np.arctan2(sin_hour, cos_hour) + np.pi) / (2 * np.pi))
hours = np.concatenate((hours, hours[-1] + np.arange(1, 7))) % 24
hours = [str(int(hour)) + 'h' for hour in hours]

sns.lineplot(
  data = pd.DataFrame({
    'hour': hours,
    'temperature': training_dataset[i][0][5,:,0].tolist() + [None] * 6,
    'humidity': training_dataset[i][0][8,:,0].tolist() + [None] * 6,
    'available bikes': (
      training_dataset[i][0][0,:,0].tolist() +
      training_dataset[i][2][0,:,0].tolist()
    )
  }).set_index('hour')
)

#%% Initialize model -----------------------------------------------------------

model = MultivariateTransformer(
  sequence_count = 11,
  sequence_width = 2,
  constant_count = 2,
  constant_width = 3,
  sequence_length = 24,
  embedding_width = 32,
  attention_width = 32,
  feedforward_width = 64,
  head_count = 8,
  layer_count = 4,
)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

#%% Train model ----------------------------------------------------------------

training_error, validation_error = train(
  training_dataset,
  model,
  optimizer,
  torch.nn.L1Loss(),
  batch_size = 16,
  epoch_count = 10 # Lower the number of epochs for shorter training
)

#%% Plot training and validation error ------------------------------------------

sns.lineplot(
  data = pd.DataFrame({
    'epochs': np.arange(1, 11),
    'training error': training_error,
    'validation error': validation_error
  }).set_index('epochs')
)

#%% Test model -----------------------------------------------------------------

(
  source_sequences,
  source_constants,
  target_sequences,
  output_sequences,
  testing_error
) = test(testing_dataset, model, torch.nn.L1Loss(), batch_size = 16)

print(testing_error)

#%% Plot prediction ------------------------------------------------------------

i = 10

sin_hour = source_sequences[i][3,:,0].numpy()
cos_hour = source_sequences[i][3,:,1].numpy()
hours = np.round(24 * (np.arctan2(sin_hour, cos_hour) + np.pi) / (2 * np.pi))
hours = np.concatenate((hours, hours[-1] + np.arange(1, 7))) % 24
hours = [str(int(hour)) + 'h' for hour in hours]

sns.lineplot(
  data = pd.DataFrame({
    'hour': hours,
    'temperature': source_sequences[i][5,:,0].tolist() + [None] * 6,
    'humidity': source_sequences[i][8,:,0].tolist() + [None] * 6,
    'available bikes': source_sequences[i][0,:,0].tolist() + target_sequences[i][0,:,0].tolist(),
    'prediction': [None] * 17 + source_sequences[i][0,-1:,0].tolist() + output_sequences[i][0,:,0].tolist()
  }).set_index('hour')
)
