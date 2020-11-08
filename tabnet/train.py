# import tensorflow as tf
from tensorflow import keras

from .model import classifier as model

# TODO: refactorize as config
DECAY_RATE = 0.95
DECAY_STEPS = 500  # DECAY_EVERY
INITIAL_LEARNING_RATE = 0.02  # INIT_LEARNING_RATE


learning_rate = keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE, DECAY_STEPS, DECAY_RATE
)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# metrics = [keras.metrics.SparseCategoricalAccuracy()]
metrics = ['sparse_categorical_accuracy']

model.compile(
    # loss=loss,  # not needed as previously added
    optimizer=optimizer,
    metrics=metrics,
)
