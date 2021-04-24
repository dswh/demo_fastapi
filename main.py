from fastapi import FastAPI
import pickle
from model_files.data_transformation import predict_mpg
from pydantic import BaseModel


class Vehicle(BaseModel):
    cylinders: int = 4
    displacement: float = 155.0
    horsepower: float = 93.0
    weight: float = 2500.0
    acceleration: float = 15.0
    model_year: int = 80


app = FastAPI()


@app.get("/")
def root():
    return {'message': "Hello World!"}


@app.post("/predict")
def predict(vehicle: Vehicle):
    '''
    This is the main prediction function which returns
    mileage of a vehicle.
    '''
    vehicle_config = {
        'Cylinder': [vehicle.cylinders],
        'Displacement': [vehicle.displacement],
        'Horsepower': [vehicle.horsepower],
        'Weight': [vehicle.weight],
        'Acceleration': [vehicle.acceleration],
        'Model Year': [vehicle.model_year],
    }

    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    prediction = predict_mpg(vehicle_config, model)

    result = {'mpg_prediction': list(prediction)}
    return result



