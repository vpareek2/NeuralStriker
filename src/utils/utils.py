import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.optimizers import Adam

def import_model():
    return load_model('model.h5')

def get_hyperparameters():
    return {
        'learning_rate': 1e-3,
        'layer1': 128,
        'layer2': 64,
        'layer3': 1,
        'activation': 'relu',
        'loss': 'mean_squared_error',
        'metrics': 'mean_absolute_error',
        'epochs': 18
    }


def get_model(hyperparameters, input_shape):
    # Extract hyperparameters
    layer1 = hyperparameters['layer1']
    layer2 = hyperparameters['layer2']
    layer3 = hyperparameters['layer3']
    activation = hyperparameters['activation']

    # Define the model
    model = Sequential([
        Dense(layer1, activation=activation),
        Dense(layer2, activation=activation),
        Dense(layer3, activation='linear')
    ])

    return model



def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_reference_data():
    return ['player_id', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age']
