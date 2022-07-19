import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras import layers, optimizers, regularizers

def get_model(vocab_len):
    "Get model based on vocab_length"
    inputs = tf.keras.Input(shape=(vocab_len,), name='Input')
    x = layers.Dropout(0.10)(inputs)
    x = layers.Dense(15, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    predictions = layers.Dense(1, activation="sigmoid",)(x)
    model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    opt = optimizers.Adam(learning_rate=0.002)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
