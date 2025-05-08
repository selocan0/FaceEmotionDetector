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

# ----------------------------------------
# dependency configurations

tf_version = package_utils.get_tf_major_version()
# ----------------------------------------

# Gender labels
labels = ["Woman", "Man"]

class GenderClient(Demography):
    """
    Gender model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Gender"

    def predict(self, img: np.ndarray) -> np.ndarray:
        if isinstance(img, tuple):
            img = img[0]

        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError(f"{self.model_name}Client: Input image is empty or has invalid shape.")

        return self.model.predict(img, verbose=0)[0, :]


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
) -> Model:
    """
    Construct gender model, download its weights, and load them.
    Returns:
        gender_model (Model)
    """

    base_model = VGGFace.base_model()

    classes = 2
    x = Convolution2D(classes, (1, 1), name="predictions")(base_model.layers[-4].output)
    x = Flatten()(x)
    x = Activation("softmax")(x)

    gender_model = Model(inputs=base_model.input, outputs=x)

    # Load weights
    home = folder_utils.get_deepface_home()
    output = os.path.join(home, ".deepface/weights/gender_model_weights.h5")

    if not os.path.isfile(output):
        logger.info(f"{os.path.basename(output)} will be downloaded...")
        gdown.download(url, output, quiet=False)

    gender_model.load_weights(output)

    # Warm-up prediction to avoid uninitialized layer issues
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    gender_model.predict(dummy_input)

    return gender_model
