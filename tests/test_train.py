import os
import joblib
from src.train import train_model

def test_model_training():
    train_model()
    os.path.exists("models/model.pkl"),"model file not founf"
    model=joblib.load("models/model.pkl")
    assert hasattr(model,"predict"),"model does nothave the predict method"
 