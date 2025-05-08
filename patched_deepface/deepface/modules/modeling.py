# built-in dependencies
from typing import Any

# project dependencies
from deepface.models.facial_recognition import (
    VGGFace,
    OpenFace,
    FbDeepFace,
    DeepID,
    ArcFace,
    SFace,
    Dlib,
    Facenet,
    GhostFaceNet,
)
from deepface.models.face_detection import (
    FastMtCnn,
    MediaPipe,
    MtCnn,
    OpenCv,
    Dlib as DlibDetector,
    RetinaFace,
    Ssd,
    Yolo,
    YuNet,
    CenterFace,
)
from deepface.models.demography import Age, Gender, Race, Emotion
from deepface.models.spoofing import FasNet

# ----------------------------------------
# Singleton cache must be defined here
cached_models = {
    "facial_recognition": {},
    "spoofing": {},
    "facial_attribute": {},
    "face_detector": {},
}
# ----------------------------------------

def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained model as a singleton.

    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier

    Returns:
        The built model class instance
    """

    models = {
        "facial_recognition": {
            "VGG-Face": VGGFace.VggFaceClient,
            "OpenFace": OpenFace.OpenFaceClient,
            "Facenet": Facenet.FaceNet128dClient,
            "Facenet512": Facenet.FaceNet512dClient,
            "DeepFace": FbDeepFace.DeepFaceClient,
            "DeepID": DeepID.DeepIdClient,
            "Dlib": Dlib.DlibClient,
            "ArcFace": ArcFace.ArcFaceClient,
            "SFace": SFace.SFaceClient,
            "GhostFaceNet": GhostFaceNet.GhostFaceNetClient,
        },
        "spoofing": {
            "Fasnet": FasNet.Fasnet,
        },
        "facial_attribute": {
            "Emotion": Emotion.EmotionClient,
            "Age": Age.ApparentAgeClient,
            "Gender": Gender.GenderClient,
            "Race": Race.RaceClient,
        },
        "face_detector": {
            "opencv": OpenCv.OpenCvClient,
            "mtcnn": MtCnn.MtCnnClient,
            "ssd": Ssd.SsdClient,
            "dlib": DlibDetector.DlibClient,
            "retinaface": RetinaFace.RetinaFaceClient,
            "mediapipe": MediaPipe.MediaPipeClient,
            "yolov8": Yolo.YoloClient,
            "yunet": YuNet.YuNetClient,
            "fastmtcnn": FastMtCnn.FastMtCnnClient,
            "centerface": CenterFace.CenterFaceClient,
        },
    }

    if task not in models:
        raise ValueError(f"Unimplemented task: {task}")

    if model_name not in models[task]:
        raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    if model_name not in cached_models[task]:
        model_class = models[task][model_name]
        cached_models[task][model_name] = model_class()

    model_instance = cached_models[task][model_name]

    print(f"✅ [DEBUG] Loaded model: {model_name}")
    print(f"    Model object: {model_instance}")
    print(f"    Model type: {type(model_instance.model)}")

    return model_instance
