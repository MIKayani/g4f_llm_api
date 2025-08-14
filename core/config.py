import json

models_list_path = '/home/itek/g4f/core/models_list.json'
models_providers_path = '/home/itek/g4f/core/models_providers.json'

def get_models_list():
    try:
        with open(models_list_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_models_providers():
    try:
        with open(models_providers_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
