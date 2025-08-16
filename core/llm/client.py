import g4f
import json
import os
from .models import get_llm_models

class LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.blacklist = self._load_blacklist()
        self.models = get_llm_models(self.blacklist)
        self.working_providers = self._get_working_providers(self.blacklist)
        
        tmp_dir = 'core_tmp'
        # Create the temporary directory if it doesn't exist
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.failed_providers_file = os.path.join(tmp_dir, 'failed_llm_providers.json')
        self.failed_providers = self._load_failed_providers()

    def _load_blacklist(self):
        blacklist_path = os.path.join('core_tmp', 'blacklist.json')
        if os.path.exists(blacklist_path):
            with open(blacklist_path, 'r') as f:
                return json.load(f)
        else:
            default_blacklist = {"blacklisted_models": [], "blacklisted_providers": [], "blacklisted_model_providers": {}}
            with open(blacklist_path, 'w') as f:
                json.dump(default_blacklist, f, indent=4)
            return default_blacklist

    def _get_working_providers(self, blacklist):
        providers = {}
        blacklisted_providers = blacklist.get('blacklisted_providers', [])
        for provider in g4f.Provider.__providers__:
            if provider.working and provider.supports_stream and provider.__name__ not in blacklisted_providers:
                providers[provider.__name__] = provider
        return providers

    def _load_failed_providers(self):
        if os.path.exists(self.failed_providers_file):
            with open(self.failed_providers_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_failed_providers(self):
        with open(self.failed_providers_file, 'w') as f:
            json.dump(self.failed_providers, f, indent=4)

    def chat(self, prompt):
        if self.model_name not in self.models:
            print(f"Model '{self.model_name}' not found.")
            return

        for model_variation in self.models[self.model_name]:
            for provider_name, provider in self.working_providers.items():
                if provider_name in self.failed_providers.get(model_variation, []):
                    continue
                
                if provider_name in self.blacklist.get('blacklisted_model_providers', {}).get(model_variation, []):
                    continue

                try:
                    print(f"Trying model: {model_variation}, provider: {provider_name}")
                    response = g4f.ChatCompletion.create(
                        model=model_variation,
                        provider=provider,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if response:
                        print("Response received:")
                        return response

                except Exception as e:
                    print(f"Provider {provider_name} failed for model {model_variation}: {e}")
                    if model_variation not in self.failed_providers:
                        self.failed_providers[model_variation] = []
                    self.failed_providers[model_variation].append(provider_name)
                    self._save_failed_providers()

        print("All providers failed for the selected model.")
        return None
