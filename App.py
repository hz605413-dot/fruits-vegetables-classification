#cd "C:\Users\hz605\OneDrive\Desktop\ML PROJECT"
#py App.py          py -3.12 App.py
import os
import uuid
import traceback

# ==================================================
# FORCE TENSORFLOW TO USE CPU ONLY
# ==================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ==================================================
# IMPORT LIBRARIES
# ==================================================

from flask import Flask, render_template, request, url_for

import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import mobilenet_v2

from werkzeug.utils import secure_filename


# ==================================================
# BASE DIRECTORY
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fruit_vegetable_model.keras"
)


print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": mobilenet_v2.preprocess_input
    },
    compile=False
)

print("Model loaded successfully!")


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
# ALLOWED IMAGE EXTENSIONS
# ==================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "webp"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
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

        print("POST request received.")

        image = request.files.get("image")


        # ==========================================
        # CHECK IF IMAGE EXISTS
        # ==========================================

        if not image or image.filename == "":

            error = (
                "Please select an image before "
                "clicking Analyze Image."
            )


        # ==========================================
        # CHECK FILE FORMAT
        # ==========================================

        elif not allowed_file(image.filename):

            error = (
                "Invalid image format. "
                "Please upload JPG, JPEG, PNG, "
                "WEBP, BMP, or GIF."
            )


        else:

            try:

                print(
                    f"Processing image: {image.filename}"
                )


                # ==================================
                # CREATE SAFE UNIQUE FILENAME
                # ==================================

                original_filename = secure_filename(
                    image.filename
                )

                unique_filename = (
                    f"{uuid.uuid4().hex}_"
                    f"{original_filename}"
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

                print(
                    f"Image saved successfully: "
                    f"{file_path}"
                )


                # ==================================
                # URL SENT TO HTML
                # ==================================

                image_path = url_for(
                    "static",
                    filename=(
                        f"uploads/{unique_filename}"
                    )
                )


                # ==================================
                # LOAD IMAGE
                # ==================================

                print("Loading image for prediction...")

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


                print(
                    f"Image shape: {img_array.shape}"
                )


                # ==================================
                # MAKE PREDICTION
                # ==================================

                print(
                    "Running model prediction..."
                )

                predictions = model.predict(
                    img_array,
                    verbose=0
                )


                print(
                    f"Raw prediction shape: "
                    f"{predictions.shape}"
                )


                print(
                    "Prediction completed."
                )


                # ==================================
                # GET BEST PREDICTION
                # ==================================

                predicted_index = int(
                    np.argmax(
                        predictions[0]
                    )
                )


                print(
                    f"Predicted index: "
                    f"{predicted_index}"
                )


                # ==================================
                # SAFETY CHECK
                # ==================================

                if (
                    predicted_index < 0
                    or predicted_index >= len(class_names)
                ):

                    raise ValueError(
                        "The model returned an invalid "
                        "class index."
                    )


                # ==================================
                # GET PREDICTION NAME
                # ==================================

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


                print(
                    f"Prediction: {prediction}"
                )

                print(
                    f"Confidence: "
                    f"{confidence:.2f}%"
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


                    # Additional safety check

                    if (
                        index >= 0
                        and index < len(class_names)
                    ):

                        top_predictions.append(
                            {
                                "name":
                                    class_names[index],

                                "confidence":
                                    round(
                                        float(
                                            predictions[0][
                                                index
                                            ] * 100
                                        ),
                                        2
                                    )
                            }
                        )


                print(
                    "Top predictions:"
                )

                for item in top_predictions:

                    print(
                        f"{item['name']} - "
                        f"{item['confidence']}%"
                    )


            # ======================================
            # ERROR HANDLING
            # ======================================

            except Exception as e:

                print(
                    "\n"
                    "===================================="
                )

                print(
                    "PREDICTION ERROR"
                )

                print(
                    "===================================="
                )

                print(
                    f"Error: {str(e)}"
                )

                traceback.print_exc()

                print(
                    "====================================\n"
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
# RUN FLASK APPLICATION LOCALLY
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )