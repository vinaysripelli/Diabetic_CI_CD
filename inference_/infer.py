from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import os

app = FastAPI()

class Item(BaseModel):
    age: int
    sex: str
    blood_pressure: str
    cholesterol: str
    na_to_k_ratio: float
    
def predict_drug(file_path,age, sex, blood_pressure, cholesterol, na_to_k_ratio):

    with open(file_path, 'rb') as f:
        pipe = pickle.load(f)

    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]

    label = f"Predicted Drug: {predicted_drug}"
    return label


@app.post("/items/")
async def create_item(item: Item):
    file_path = '../Model/drug_pipeline.pkl'

    if os.path.isfile(file_path):
        return predict_drug(file_path,item.age,item.sex,item.blood_pressure,item.cholesterol,item.na_to_k_ratio)
    else:
        return ("Training not finished.")
    
@app.get("/")
async def main():
    return {"message": "Hello World"}