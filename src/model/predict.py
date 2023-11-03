import pandas as pd
from tensorflow.keras.models import load_model

from utils.utils import get_reference_data, import_model


df = pd.read_csv('../../data/final/2023.csv').drop(columns=['Unnamed: 0'])

# Assuming import_model is correctly implemented to load the desired model
model = import_model()

reference_data = get_reference_data(df)
X = df.drop(columns=reference_data + ['Gls', 'Ast', 'xAG', 'npxG+xAG', 'npxG'])

predictions = model.predict(X)

results = pd.concat([df[reference_data], pd.DataFrame(predictions, columns=['Predicted_Gls'])], axis=1)
results['Predicted_Gls'] = results['Predicted_Gls'].apply(lambda x: max(0, x))

results.to_csv('../../data/final/predictions.csv', index=False)
