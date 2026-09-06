#cd "C:\Users\hz605\OneDrive\Desktop\ML PROJECT"
#py App.py          py -3.12 App.py
# ==================================================
# App.py
# ==================================================

import os
import uuid
import traceback


# ==================================================
# FORCE TENSORFLOW TO USE CPU ONLY
# IMPORTANT: MUST BE BEFORE importing TensorFlow
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
# TENSORFLOW CPU SETTINGS
# ==================================================

try:

    tf.config.threading.set_intra_op_parallelism_threads(1)

    tf.config.threading.set_inter_op_parallelism_threads(1)

except Exception:

    pass


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


print("Loading trained model...", flush=True)


model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "preprocess_input":
            mobilenet_v2.preprocess_input
    },

    compile=False
)


print(
    "Model loaded successfully!",
    flush=True
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


app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# Optional upload size limit
app.config[
    "MAX_CONTENT_LENGTH"
] = 10 * 1024 * 1024


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


# ==================================================
# CHECK ALLOWED FILE
# ==================================================

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()

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

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)


def home():


    # ==============================================
    # DEFAULT VALUES
    # ==============================================

    prediction = None

    confidence = None

    image_path = None

    top_predictions = []

    error = None


    # ==============================================
    # WHEN USER UPLOADS AN IMAGE
    # ==============================================

    if request.method == "POST":


        print(
            "POST request received.",
            flush=True
        )


        image = request.files.get(
            "image"
        )


        # ==========================================
        # CHECK IF IMAGE EXISTS
        # ==========================================

        if (

            not image

            or

            image.filename == ""

        ):


            error = (

                "Please select an image before "
                "clicking Analyze Image."

            )


        # ==========================================
        # CHECK FILE FORMAT
        # ==========================================

        elif not allowed_file(
            image.filename
        ):


            error = (

                "Invalid image format. "
                "Please upload JPG, JPEG, PNG, "
                "WEBP, BMP, or GIF."

            )


        # ==========================================
        # PROCESS IMAGE
        # ==========================================

        else:


            try:


                print(
                    f"Processing image: "
                    f"{image.filename}",
                    flush=True
                )


                # ==================================
                # CREATE SAFE UNIQUE FILENAME
                # ==================================

                original_filename = (
                    secure_filename(
                        image.filename
                    )
                )


                unique_filename = (

                    f"{uuid.uuid4().hex}_"
                    f"{original_filename}"

                )


                # ==================================
                # SAVE IMAGE
                # ==================================

                file_path = os.path.join(

                    app.config[
                        "UPLOAD_FOLDER"
                    ],

                    unique_filename

                )


                image.save(
                    file_path
                )


                print(
                    f"Image saved: "
                    f"{file_path}",
                    flush=True
                )


                # ==================================
                # URL SENT TO HTML
                # ==================================

                image_path = url_for(

                    "static",

                    filename=(
                        f"uploads/"
                        f"{unique_filename}"
                    )

                )


                # ==================================
                # LOAD IMAGE
                # ==================================

                print(
                    "Loading image for prediction...",
                    flush=True
                )


                img = load_img(

                    file_path,

                    target_size=(
                        128,
                        128
                    )

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


                # Make sure TensorFlow receives
                # float32 data

                img_array = img_array.astype(
                    np.float32
                )


                print(
                    f"Image shape: "
                    f"{img_array.shape}",
                    flush=True
                )


                # ==================================
                # MAKE PREDICTION
                # ==================================

                print(
                    "Running model prediction...",
                    flush=True
                )


                # IMPORTANT:
                # Use direct inference instead of
                # model.predict()

                predictions = model(

                    img_array,

                    training=False

                ).numpy()


                print(
                    f"Raw prediction shape: "
                    f"{predictions.shape}",
                    flush=True
                )


                print(
                    "Prediction completed.",
                    flush=True
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
                    f"{predicted_index}",
                    flush=True
                )


                # ==================================
                # SAFETY CHECK
                # ==================================

                if (

                    predicted_index < 0

                    or

                    predicted_index >= len(
                        class_names
                    )

                ):


                    raise ValueError(

                        "The model returned an "
                        "invalid class index."

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
                    ]

                    * 100

                )


                print(
                    f"Prediction: "
                    f"{prediction}",
                    flush=True
                )


                print(
                    f"Confidence: "
                    f"{confidence:.2f}%",
                    flush=True
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


                    index = int(
                        index
                    )


                    if (

                        index >= 0

                        and

                        index < len(
                            class_names
                        )

                    ):


                        top_predictions.append(

                            {

                                "name":
                                    class_names[
                                        index
                                    ],

                                "confidence":

                                    round(

                                        float(

                                            predictions[0][
                                                index
                                            ]

                                            * 100

                                        ),

                                        2

                                    )

                            }

                        )


                # ==================================
                # PRINT TOP 3 RESULTS
                # ==================================

                print(
                    "Top predictions:",
                    flush=True
                )


                for item in top_predictions:


                    print(

                        f"{item['name']} - "
                        f"{item['confidence']}%",

                        flush=True

                    )


            # ======================================
            # ERROR HANDLING
            # ======================================

            except Exception as e:


                print(
                    "\n"
                    "====================================",
                    flush=True
                )


                print(
                    "PREDICTION ERROR",
                    flush=True
                )


                print(
                    "====================================",
                    flush=True
                )


                print(
                    f"Error: {str(e)}",
                    flush=True
                )


                traceback.print_exc()


                print(
                    "====================================\n",
                    flush=True
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