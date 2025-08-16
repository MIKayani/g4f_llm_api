import asyncio
import json
import os
from g4f.client import Client
from .models import get_image_models

class ImageGen:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = Client()
        self.blacklist = self._load_blacklist()
        self.models = get_image_models(self.blacklist)

    def _load_blacklist(self):
        tmp_dir = 'core_tmp'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        blacklist_path = os.path.join(tmp_dir, 'blacklist.json')
        if os.path.exists(blacklist_path):
            with open(blacklist_path, 'r') as f:
                return json.load(f)
        else:
            default_blacklist = {"blacklisted_models": [], "blacklisted_providers": [], "blacklisted_model_providers": {}}
            with open(blacklist_path, 'w') as f:
                json.dump(default_blacklist, f, indent=4)
            return default_blacklist

    async def generate(self, prompt):
        if self.model_name not in self.models:
            print(f"Model '{self.model_name}' not found.")
            return None

        for model_variation in self.models[self.model_name]:
            try:
                print(f"Trying model variation: {model_variation}...")
                response = await self.client.images.async_generate(
                    model=model_variation,
                    prompt=prompt,
                    response_format="url"
                )
                if response.data and response.data[0].url:
                    print("Image generated successfully!")
                    return response.data[0].url
            except Exception as e:
                print(f"Model variation '{model_variation}' failed: {e}")
        
        print("All model variations failed.")
        return None
