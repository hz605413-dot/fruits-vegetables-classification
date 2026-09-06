#cd "C:\Users\hz605\OneDrive\Desktop\ML PROJECT"
#py App.py          py -3.12 App.py

from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import mobilenet_v2
from werkzeug.utils import secure_filename


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

MODEL_PATH = "fruit_vegetable_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": mobilenet_v2.preprocess_input
    },
    compile=False
)


# ==================================================
# CREATE FLASK APPLICATION
# ==================================================

app = Flask(__name__)


# ==================================================
# UPLOAD FOLDER
# ==================================================

UPLOAD_FOLDER = os.path.join("static", "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ==================================================
# CLASS NAMES
# MUST MATCH GOOGLE COLAB TRAINING ORDER EXACTLY
# ==================================================

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


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None
    top_predictions = []

    # ==============================================
    # WHEN USER UPLOADS AN IMAGE
    # ==============================================

    if request.method == "POST":

        image = request.files.get("image")

        # ==========================================
        # CHECK IMAGE
        # ==========================================

        if image and image.filename != "":

            # ======================================
            # SAVE IMAGE
            # ======================================

            filename = secure_filename(
                image.filename
            )

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(
                file_path
            )

            # Path sent to HTML
            image_path = file_path


            # ======================================
            # LOAD IMAGE
            # ======================================

            img = load_img(
                file_path,
                target_size=(128, 128)
            )


            # ======================================
            # CONVERT IMAGE TO NUMPY ARRAY
            # ======================================

            img_array = img_to_array(
                img
            )


            # ======================================
            # ADD BATCH DIMENSION
            # ======================================

            img_array = np.expand_dims(
                img_array,
                axis=0
            )


            # ======================================
            # IMPORTANT:
            # DO NOT APPLY preprocess_input HERE.
            #
            # Your saved model already contains the
            # MobileNetV2 preprocessing Lambda layer.
            # ======================================


            # ======================================
            # MAKE PREDICTION
            # ======================================

            predictions = model.predict(
                img_array,
                verbose=0
            )


            # ======================================
            # GET BEST PREDICTION
            # ======================================

            predicted_index = int(
                np.argmax(
                    predictions[0]
                )
            )


            prediction = class_names[
                predicted_index
            ]


            # ======================================
            # CONFIDENCE PERCENTAGE
            # ======================================

            confidence = float(
                predictions[0][
                    predicted_index
                ] * 100
            )


            # ======================================
            # GET TOP 3 PREDICTIONS
            # ======================================

            top_indices = np.argsort(
                predictions[0]
            )[-3:][::-1]


            # ======================================
            # SAVE TOP 3 RESULTS
            # ======================================

            for index in top_indices:

                top_predictions.append(
                    {
                        "name": class_names[
                            int(index)
                        ],

                        "confidence": round(
                            float(
                                predictions[0][
                                    index
                                ] * 100
                            ),
                            2
                        )
                    }
                )


    # ==================================================
    # SEND DATA TO HTML
    # ==================================================

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        top_predictions=top_predictions
    )


# ==================================================
# RUN FLASK APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )