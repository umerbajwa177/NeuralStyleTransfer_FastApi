from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import cv2
import tensorflow as tf #tensorflow
from matplotlib import pyplot as plt 
import numpy as np

app = FastAPI()

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    content_image = load_image(file_location)
    model = tf.keras.models.load_model('Saved_Model/')

    style_image = load_image(r"C:\Users\Acer\OneDrive\Desktop\modelapi\s (6).jpg")

    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    # Saving the generated image
    plt.imshow(np.squeeze(stylized_image))
    plt.show()

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')