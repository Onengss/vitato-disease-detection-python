
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

MODEL = tf.keras.models.load_model("../saved_models/1.keras")
CLASS_NAMES = ["Caterpillar", "Healthy", "Infacted", "Leaf_Miners", "Sap_Sucking_Insect"]

@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }

    return


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)