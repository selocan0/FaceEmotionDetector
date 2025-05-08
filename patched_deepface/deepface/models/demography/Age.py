import os
import gdown
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, Flatten, Activation
from deepface.models.facial_recognition import VGGFace
from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger

logger = Logger()

# --------------------
# Labels and version
tf_version = package_utils.get_tf_major_version()
# --------------------

class ApparentAgeClient(Demography):
    """
    Age model class.
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Age"

    def predict(self, img: np.ndarray) -> np.float64:
        if isinstance(img, tuple):
            img = img[0]

        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("ApparentAgeClient: Input image is empty or has invalid shape.")

        age_predictions = self.model.predict(img, verbose=0)[0, :]
        return find_apparent_age(age_predictions)


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
) -> Model:
    """
    Construct age model, download its weights and load.
    Returns:
        age_model (Model)
    """

    base_model = VGGFace.base_model()

    classes = 101
    x = Convolution2D(classes, (1, 1), name="predictions")(base_model.layers[-4].output)
    x = Flatten()(x)
    x = Activation("softmax")(x)

    age_model = Model(inputs=base_model.input, outputs=x)

    # Load weights
    home = folder_utils.get_deepface_home()
    output = os.path.join(home, ".deepface/weights/age_model_weights.h5")

    if not os.path.isfile(output):
        logger.info(f"{os.path.basename(output)} will be downloaded...")
        gdown.download(url, output, quiet=False)

    age_model.load_weights(output)

    # Warm-up call
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    age_model.predict(dummy_input)

    return age_model

def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
    """
    Calculate apparent age from softmax distribution.
    """
    output_indexes = np.arange(0, 101)
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age
