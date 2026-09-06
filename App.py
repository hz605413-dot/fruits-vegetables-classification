#cd "C:\Users\hz605\OneDrive\Desktop\ML PROJECT"
#py App.py          py -3.12 App.py
import os

# Force TensorFlow to use CPU only on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import mobilenet_v2
from werkzeug.utils import secure_filename



# ==================================================
# BASE DIRECTORY
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fruit_vegetable_model.keras"
)

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

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

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
    error = None


    # ==============================================
    # WHEN USER UPLOADS AN IMAGE
    # ==============================================

    if request.method == "POST":

        image = request.files.get("image")


        # ==========================================
        # CHECK IMAGE
        # ==========================================

        if not image or image.filename == "":

            error = "Please select an image before clicking Analyze Image."

        else:

            try:

                # ==================================
                # CREATE SAFE UNIQUE FILENAME
                # ==================================

                original_filename = secure_filename(
                    image.filename
                )

                unique_filename = (
                    f"{uuid.uuid4().hex}_{original_filename}"
                )


                # ==================================
                # SAVE IMAGE
                # ==================================

                file_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_filename
                )

                image.save(
                    file_path
                )


                # ==================================
                # URL SENT TO HTML
                # ==================================

                image_path = url_for(
                    "static",
                    filename=f"uploads/{unique_filename}"
                )


                # ==================================
                # LOAD IMAGE
                # ==================================

                img = load_img(
                    file_path,
                    target_size=(128, 128)
                )


                # ==================================
                # CONVERT IMAGE TO NUMPY ARRAY
                # ==================================

                img_array = img_to_array(
                    img
                )


                # ==================================
                # ADD BATCH DIMENSION
                # ==================================

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )


                # ==================================
                # IMPORTANT:
                #
                # DO NOT APPLY preprocess_input HERE.
                #
                # Your saved model already contains
                # the MobileNetV2 preprocessing
                # Lambda layer.
                # ==================================


                # ==================================
                # MAKE PREDICTION
                # ==================================

                predictions = model.predict(
                    img_array,
                    verbose=0
                )


                # ==================================
                # GET BEST PREDICTION
                # ==================================

                predicted_index = int(
                    np.argmax(
                        predictions[0]
                    )
                )


                # Safety check
                if predicted_index >= len(class_names):

                    raise ValueError(
                        "The model returned an invalid class index."
                    )


                prediction = class_names[
                    predicted_index
                ]


                # ==================================
                # CONFIDENCE PERCENTAGE
                # ==================================

                confidence = float(
                    predictions[0][
                        predicted_index
                    ] * 100
                )


                # ==================================
                # GET TOP 3 PREDICTIONS
                # ==================================

                top_indices = np.argsort(
                    predictions[0]
                )[-3:][::-1]


                # ==================================
                # SAVE TOP 3 RESULTS
                # ==================================

                for index in top_indices:

                    index = int(index)

                    top_predictions.append(
                        {
                            "name": class_names[
                                index
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


            except Exception as e:

                print(
                    f"Prediction error: {e}"
                )

                error = (
                    "Unable to analyze this image. "
                    "Please try another image."
                )


    # ==================================================
    # SEND DATA TO HTML
    # ==================================================

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        top_predictions=top_predictions,
        error=error
    )


# ==================================================
# RUN FLASK APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )