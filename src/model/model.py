import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the hyperparameters for the model. These are the settings that can be tuned to optimize performance.
learning_rate = 1e-3
layer1 = 128
layer2 = 64
layer3 = 1
activation = 'relu'
loss = 'mean_squared_error'
metrics = 'mean_absolute_error'
epochs = 18

# Load the dataset from a CSV file and drop the first unnamed column which usually is an index column.
df = pd.read_csv('../../data/final/2022.csv').drop(columns=['Unnamed: 0'])
# List of columns in the dataframe that are not features for training.
reference_data = ['player_id', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age']
# Features for training (all columns except reference_data and targets)
X = df.drop(columns=reference_data + ['Gls', 'Ast', 'xAG', 'npxG+xAG', 'npxG'])
# Target variable for prediction
y = df['Gls']

# Split the dataset into training, cross-validation, and test sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# Initialize the Sequential model which is a linear stack of layers.
model = Sequential([
    Dense(layer1, activation=activation), # First hidden layer with 'relu' activation
    Dense(layer2, activation=activation), # Second hidden layer with 'relu' activation
    Dense(layer3, activation='linear')    # Output layer with linear activation
])

# Compile the model with the Adam optimizer, a specified loss function, and a metric for performance evaluation.
model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss=loss,
              metrics=metrics)

# Fit the model to the training data with a certain number of epochs, with early stopping based on cross-validation loss.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_cv, y_cv), callbacks=[early_stop], verbose=1)

# Save the trained model to disk for future use.
model.save('model.h5')
