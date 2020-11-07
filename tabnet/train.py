# import tensorflow as tf
from tensorflow import keras

from .model import classifier as model

# TODO: refactorize as config
DECAY_RATE = 0.95
DECAY_STEPS = 500  # DECAY_EVERY
INITIAL_LEARNING_RATE = 0.02  # INIT_LEARNING_RATE


# TODO: customize loss with total_entropy
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

learning_rate = keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE, DECAY_STEPS, DECAY_RATE
)
optimizer = keras.optimizer.AdamOptimizer(learning_rate=learning_rate)

metrics = [keras.metrics.SparseCategoricalAccuracy()]

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
)
