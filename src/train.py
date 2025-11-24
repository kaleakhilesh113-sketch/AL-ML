
    
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2.2, 4.1, 6.0, 8.1, 10.2]
    })

    x = data[['x']]
    y = data['y']

    model = LinearRegression()
    model.fit(x, y)

    # ✅ Ensure 'models' directory exists before saving
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, "models/model.pkl")

# Run the function
if __name__== "__main__":
    train_model()