import os
import gdown
import numpy as np
import cv2

from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

logger = Logger()

# -------------------------------
# Dependency configuration
tf_version = package_utils.get_tf_major_version()

# Labels for the emotions the model can detect
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# -------------------------------
# Emotion Model Class
class EmotionClient(Demography):
    """
    Emotion model client
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        if isinstance(img, tuple):
            img = img[0]

        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("EmotionClient: Input image is empty or has invalid shape.")

        if len(img.shape) == 3 and img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=-1)
        img_gray = np.expand_dims(img_gray, axis=0)

        emotion_predictions = self.model.predict(img_gray, verbose=0)[0, :]
        return emotion_predictions


# -------------------------------
# Load Emotion Model
def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
) -> Model:
    """
    Construct and load the emotion model
    """

    print("✅ [DEBUG] Emotion model loading...")

    num_classes = 7
    input_layer = Input(shape=(48, 48, 1))

    x = Conv2D(64, (5, 5), activation="relu")(input_layer)
    x = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Load weights
    home = folder_utils.get_deepface_home()
    output_path = os.path.join(home, ".deepface/weights/facial_expression_model_weights.h5")

    if not os.path.isfile(output_path):
        logger.info(f"{os.path.basename(output_path)} will be downloaded...")
        gdown.download(url, output_path, quiet=False)

    model.load_weights(output_path)

    # Warm-up prediction to ensure layers are initialized
    dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
    _ = model.predict(dummy_input)

    print("✅ [DEBUG] Emotion model ready.")
    return model
