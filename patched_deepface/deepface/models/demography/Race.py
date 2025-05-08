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

# --------------------------
# Race labels
labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

# --------------------------
class RaceClient(Demography):
    """
    Race model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "Race"

    def predict(self, img: np.ndarray) -> np.ndarray:
        if isinstance(img, tuple):
            img = img[0]

        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError(f"{self.model_name}Client: Input image is empty or has invalid shape.")

        return self.model.predict(img, verbose=0)[0, :]


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
) -> Model:
    """
    Construct race model, download its weights and load them.
    Returns:
        race_model (Model)
    """

    base = VGGFace.base_model()

    classes = 6
    x = Convolution2D(classes, (1, 1), name="predictions")(base.layers[-4].output)
    x = Flatten()(x)
    x = Activation("softmax")(x)

    race_model = Model(inputs=base.input, outputs=x)

    # Download and load weights
    home = folder_utils.get_deepface_home()
    output = os.path.join(home, ".deepface/weights/race_model_single_batch.h5")

    if not os.path.isfile(output):
        logger.info(f"{os.path.basename(output)} will be downloaded...")
        gdown.download(url, output, quiet=False)

    race_model.load_weights(output)

    # Warm-up dummy call to trigger `build`
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    race_model.predict(dummy_input)

    return race_model
