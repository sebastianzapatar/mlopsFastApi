from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from joblib import load
import numpy as np
import uvicorn
app = FastAPI()

# Cargar el modelo
model = load('modelo_entrenado.joblib')

class Item(BaseModel):
    features: list[int]

@app.post("/predict/")
def predict(item: Item):
    features_array = np.array(item.features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
