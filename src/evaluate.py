import joblib
import numpy as np

def evaluate_model():
    
    # load the trained model
    
    model=joblib.load("models/model.pkl")
    
    text_x=np.array([[6],[7],[8]])
    
    preds=model.predict(text_x)
    
    print("Predictions ",preds)
    
    return preds

if __name__=="__main__":
    evaluate_model()