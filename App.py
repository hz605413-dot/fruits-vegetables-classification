#cd "C:\Users\hz605\OneDrive\Desktop\ML PROJECT"
#py App.py          py -3.12 App.py
import os

# Reduce TensorFlow resource usage on Render
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import traceback


# ==============================
# CREATE FLASK APP
# ==============================

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==============================
# UPLOAD FOLDER
# ==============================

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
# LIMIT TENSORFLOW CPU THREADS
# ==============================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(
    "fruit_vegetable_model.keras",
    compile=False
)

print("Model loaded successfully!")


# ==============================
# CLASS NAMES
# ==============================

class_names = [
    "FreshApple",
    "FreshBanana",
    "FreshBellpepper",
    "FreshCarrot",
    "FreshCucumber",
    "FreshMango",
    "FreshOrange",
    "FreshPotato",
    "FreshStrawberry",
    "FreshTomato",
    "RottenApple",
    "RottenBanana",
    "RottenBellpepper",
    "RottenCarrot",
    "RottenCucumber",
    "RottenMango",
    "RottenOrange",
    "RottenPotato",
    "RottenStrawberry",
    "RottenTomato"
]


# ==============================
# HOME PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None
    error_message = None

    if request.method == "POST":

        try:

            print("POST request received")

            image = request.files.get("image")

            if not image or image.filename == "":
                error_message = "Please select an image."

            else:

                # Safe filename
                filename = secure_filename(image.filename)

                file_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                # Save image
                image.save(file_path)

                image_path = file_path

                print(f"Image saved: {file_path}")

                # ==============================
                # OPEN IMAGE USING PIL
                # ==============================

                img = Image.open(file_path)

                # Convert to RGB
                img = img.convert("RGB")

                # Resize
                img = img.resize((128, 128))

                # Convert to NumPy array
                img_array = np.array(
                    img,
                    dtype=np.float32
                )

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                print("Image prepared successfully")

                # ==============================
                # PREDICTION
                # ==============================

                predictions = model(
                    img_array,
                    training=False
                ).numpy()

                predicted_index = int(
                    np.argmax(predictions[0])
                )

                prediction = class_names[
                    predicted_index
                ]

                confidence = float(
                    predictions[0][predicted_index] * 100
                )

                print(
                    f"Prediction: {prediction}"
                )

                print(
                    f"Confidence: {confidence}"
                )

        except Exception as e:

            print("ERROR DURING PREDICTION:")
            print(traceback.format_exc())

            error_message = str(e)


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error_message=error_message
    )


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    app.run()
