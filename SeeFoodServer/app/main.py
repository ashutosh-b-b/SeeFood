from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import DeployedPredictor
app = FastAPI()
predictor = DeployedPredictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64Image(BaseModel):
    image: str

@app.get("/")
def root():
    return {"message": "SeeFoodServer is running"}

@app.post("/predict")
def predict(payload: Base64Image):
    label = predictor.predict_image(payload.image)
    return {"label": label}
