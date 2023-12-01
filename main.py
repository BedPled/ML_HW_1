from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pickle
import pandas as pd
from sklearn.linear_model import Ridge

categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    with open('Ridge_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('Ridge_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('Ridge_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    df = pd.DataFrame([item.dict()])

    del df['name']
    del df['torque']
    del df['selling_price'] # дропаем таргет

    # чистим
    df['mileage'] = df['mileage'].apply(lambda x: float(str(x)[:-5]) if x == x else None)
    df['engine'] = df['engine'].apply(lambda x: float(str(x)[:-3]) if x == x else None)
    df['max_power'] = df['engine'].apply(lambda x: float(str(x)[:-3]) if x == x else None)

    # заполняем пропуски
    df['mileage'] = df['mileage'].apply(lambda x: 19.33 if x != x else x)
    df['engine'] = df['engine'].apply(lambda x: 1248 if x != x else int(x))
    df['max_power'] = df['max_power'].apply(lambda x: 124 if x != x else x)
    df['seats'] = df['seats'].apply(lambda x: 5 if x != x else int(x))


    # энкодим
    encoder_df = pd.DataFrame(encoder.transform(df[categorical_features]).toarray(),
                              columns=encoder.get_feature_names_out(categorical_features))
    df = df.join(encoder_df)

    for i in categorical_features:
        del df[i]

    # скейлим
    df = scaler.transform(df)

    # predict
    pred = model.predict(df)

    return int(pred)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    with open('Ridge_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('Ridge_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('Ridge_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    df = pd.DataFrame([obj.dict() for obj in items])

    del df['name']
    del df['torque']
    del df['selling_price'] # дропаем таргет

    # чистим
    df['mileage'] = df['mileage'].apply(lambda x: float(str(x)[:-5]) if x == x else None)
    df['engine'] = df['engine'].apply(lambda x: float(str(x)[:-3]) if x == x else None)
    df['max_power'] = df['engine'].apply(lambda x: float(str(x)[:-3]) if x == x else None)

    # заполняем пропуски
    df['mileage'] = df['mileage'].apply(lambda x: 19.33 if x != x else x)
    df['engine'] = df['engine'].apply(lambda x: 1248 if x != x else int(x))
    df['max_power'] = df['max_power'].apply(lambda x: 124 if x != x else x)
    df['seats'] = df['seats'].apply(lambda x: 5 if x != x else int(x))


    # энкодим
    encoder_df = pd.DataFrame(encoder.transform(df[categorical_features]).toarray(),
                              columns=encoder.get_feature_names_out(categorical_features))
    df = df.join(encoder_df)

    for i in categorical_features:
        del df[i]

    # скейлим
    df = scaler.transform(df)

    # predict
    pred = model.predict(df)

    return list(pred.astype(int))