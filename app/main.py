from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import json


with open("model.pkl","rb") as file:
    model=pickle.load(file)

model_prediction_map={"0":"Books","1":"Clothing & Accessories","2":"Electronics","3":"Household"}
      
    
app=FastAPI()

class Review(BaseModel):
    text:str
    
@app.post("/predict/")
def predict(review:Review):
    model_prediction=model.predict(Review.text)
    return model_prediction_map[str(model_prediction[0])]

if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)