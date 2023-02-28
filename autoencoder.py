# Importing required libraries
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model

# Generating sample data
data = np.random.random((1000, 100))

# Creating autoencoder architecture
input_data = Input(shape=(100,))
encoded = Dense(32, activation='relu')(input_data)
decoded = Dense(100, activation='sigmoid')(encoded)

# Creating autoencoder model
autoencoder = Model(input_data, decoded)

# Compiling the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Training the model
autoencoder.fit(data, data, epochs=50, batch_size=32, shuffle=True)

# Extracting the encoder and decoder models
encoder = Model(input_data, encoded)
encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Testing the model
encoded_data = encoder.predict(data)
decoded_data = decoder.predict(encoded_data)
