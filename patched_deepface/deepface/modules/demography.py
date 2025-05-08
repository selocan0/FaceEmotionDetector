# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
from tqdm import tqdm
import cv2

# project dependencies
from deepface.modules import modeling, detection, preprocessing
from deepface.models.demography import Gender, Race, Emotion
from keras.models import Sequential, Model

def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:

    if isinstance(actions, str):
        actions = (actions,)

    if not hasattr(actions, "__getitem__") or not actions:
        raise ValueError("`actions` must be a list of strings.")

    actions = list(actions)

    for action in actions:
        if action not in ("emotion", "age", "gender", "race"):
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`, `age`, `gender`, `race`."
            )

    resp_objects = []

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in the given image.")

        img_content = img_obj["face"]

        if isinstance(img_content, tuple):
            print("⚠️ [DEBUG] Detected tuple in img_content, extracting first element...")
            img_content = img_content[0]

        if img_content is None or img_content.size == 0:
            raise ValueError("❌ analyze: img_content is None or empty")

        if len(img_content.shape) == 2:
            print("🔁 [DEBUG] Converting grayscale to BGR")
            img_content = cv2.cvtColor(img_content, cv2.COLOR_GRAY2BGR)
        elif img_content.shape[2] == 1:
            print("🔁 [DEBUG] Converting single-channel to BGR")
            img_content = cv2.cvtColor(img_content, cv2.COLOR_GRAY2BGR)

        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            print("❌ [DEBUG] Skipping image: detected face has zero width or height.")
            continue

        img_region = img_obj["facial_area"]
        img_confidence = img_obj["confidence"]

        img_content = img_content[:, :, ::-1]
        img_content = preprocessing.resize_image(img=img_content, target_size=(224, 224))

        obj = {}

        pbar = tqdm(
            range(0, len(actions)),
            desc="Finding actions",
            disable=silent if len(actions) > 1 else True,
        )
        for index in pbar:
            action = actions[index]
            pbar.set_description(f"Action: {action}")
            print(f"\U0001f9e0 [DEBUG] Running action: {action}")

            model = modeling.build_model("facial_attribute", action.capitalize())
            print(f"\U0001f4e6 [DEBUG] Built model: {model.model_name}, Type: {type(model)}")
            print(f"    Inner model type: {type(model.model)}")
            print(f"    Is Sequential? {isinstance(model.model, Sequential)}")
            print(f"    Is Functional? {isinstance(model.model, Model)}")

            predictions = model.predict(img_content)

            if action == "emotion":
                sum_of_predictions = predictions.sum()
                obj["emotion"] = {
                    label: 100 * predictions[i] / sum_of_predictions
                    for i, label in enumerate(Emotion.labels)
                }
                obj["dominant_emotion"] = Emotion.labels[np.argmax(predictions)]

            elif action == "age":
                obj["age"] = int(predictions)

            elif action == "gender":
                obj["gender"] = {
                    label: 100 * predictions[i]
                    for i, label in enumerate(Gender.labels)
                }
                obj["dominant_gender"] = Gender.labels[np.argmax(predictions)]

            elif action == "race":
                sum_of_predictions = predictions.sum()
                obj["race"] = {
                    label: 100 * predictions[i] / sum_of_predictions
                    for i, label in enumerate(Race.labels)
                }
                obj["dominant_race"] = Race.labels[np.argmax(predictions)]

        obj["region"] = img_region
        obj["face_confidence"] = img_confidence

        resp_objects.append(obj)

    return resp_objects
