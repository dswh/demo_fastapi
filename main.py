from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from model_files.data_transformation import predict_mpg


class Vehicle(BaseModel):
    cylinders: int = 4
    displacement: float = 155.0
    horsepower: float = 93.0
    weight: float = 2500.0
    acceleration: float = 15.0
    model_year: int = 81


app = FastAPI()

@app.get("/")
async def hello():
    return {'message': "Hello World!"}


@app.post("/predict/")
async def predict(vehicle: Vehicle):
    vehicle_config = {
        'Cylinders': [vehicle.cylinders],
        'Displacement': [vehicle.displacement],
        'Horsepower': [vehicle.horsepower],
        'Weight': [vehicle.weight],
        'Acceleration': [vehicle.acceleration],
        'Model Year': [vehicle.model_year],
    }

    with open('./model_files/model.bin', 'rb') as fin:
        model = pickle.load(fin)
        fin.close()

    prediction = predict_mpg(vehicle_config, model)
    result = {
        'mpg_prediction': list(prediction)
    }
    return result
