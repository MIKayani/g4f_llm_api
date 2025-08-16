from .llm import LLM
from .image import ImageGen

from .llm.models import get_llm_models
from .image.models import get_image_models

def llm_models_available():
    return get_llm_models().keys()

def image_models_available():
    return get_image_models().keys()