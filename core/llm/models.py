import g4f
import json
import re
from collections import defaultdict

def normalize_model_name(name):
    """
    Normalizes a model name to group similar models.
    """
    s = name.lower()
    s = s.replace('_', '-').replace(' ', '-')

    # Remove common prefixes like 'hf:', 'meta-llama/', etc.
    if ':' in s:
        s = s.split(':', 1)[-1]
    if '/' in s:
        s = s.split('/', 1)[-1]

    s = s.replace('_', '-')
    s = re.sub(r'([a-zA-Z])(\d)', r'\1-\2', s)  # insert hyphen between letters+numbers
    s = re.sub(r'(\d+)\.0\b', r'\1', s)         # remove .0 from version numbers

    # remove date-like patterns
    s = re.sub(r'[- ]\d{2}[- ]\d{2}', '', s)  # handles -06-17 or 06-17 with spaces
    s = re.sub(r'\b\d{4}\b', '', s)           # 2024
    s = re.sub(r'\b\d{6}\b', '', s)           # 202401
    s = re.sub(r'\b\d{8}\b', '', s)           # 20240101

    # Truncate after 'distill'
    distill_match = re.search(r'distill', s)
    if distill_match:
        s = s[:distill_match.end()]

    # Remove common suffixes
    s = re.sub(r'-latest|-exp|-experimental|-with-apps|-preview|-distill', '', s)
    s = re.sub(r'api|lora|image|audio|chat|experimental', '', s)

    # Cleanup: collapse multiple hyphens and strip leading/trailing
    s = re.sub(r'-+', '-', s)       # collapse repeated hyphens
    s = s.strip('- ')

    return s

def get_llm_models(blacklist=None):
    """
    Gathers all models from g4f providers, groups them by a normalized name,
    filters out smaller models, and saves the result to a JSON file.
    """
    if blacklist is None:
        blacklist = {"blacklisted_models": [], "blacklisted_providers": [], "blacklisted_model_providers": {}}
    
    model_groups = defaultdict(set)

    for provider in g4f.Provider.__providers__:
        if hasattr(provider, 'models') and provider.models:
            model_list = []
            if isinstance(provider.models, list):
                model_list = provider.models
            elif isinstance(provider.models, dict):
                model_list = list(provider.models.keys())
            elif isinstance(provider.models, set):
                model_list = list(provider.models)

            for model_name in model_list:
                if isinstance(model_name, str) and model_name:
                    normalized_name = normalize_model_name(model_name)

                    if normalized_name:
                        # Skip audio/image/coder/tts models
                        if any(x in model_name for x in ["audio", "image", "tts", "coder", "ghibli"]):
                            continue
                        # Skip non-ascii names
                        try:
                            normalized_name.encode('ascii')
                        except UnicodeEncodeError:
                            continue
                        model_groups[normalized_name].add(model_name)

    # Filter out blacklisted models
    blacklisted_models = blacklist.get('blacklisted_models', [])
    model_groups = {name: raw_names for name, raw_names in model_groups.items() if name not in blacklisted_models}

    # Filter out smaller models
    filtered_model_groups = {}
    for name, raw_names in model_groups.items():
        keep_model = True
        matches = re.finditer(r'(\d+(?:\.\d+)?)([bm])', name)
        for match in matches:
            try:
                number = float(match.group(1))
                if number < 30:
                    preceding_text = name[:match.start()]
                    words_before = preceding_text.split()
                    if not (words_before and words_before[-1].endswith('distilled')):
                        keep_model = False
                        break
            except ValueError:
                pass
        if keep_model:
            filtered_model_groups[name] = raw_names

    # Famous LLMs we keep
    # famous_llm_names = ["gemini", "gpt", "anthropic", "grok", "mistral", "llama", "deepseek", "o-", "qwen"]
    famous_llm_names = ["gemini", "gpt", "anthropic", "grok", "deepseek", "o-", "qwen"]

    final_filtered_model_groups = {}
    for name, raw_names in filtered_model_groups.items():
        for llm_name in famous_llm_names:
            if name.startswith(llm_name):
                final_filtered_model_groups[name] = raw_names
                break

    # Convert sets to sorted lists
    # final_model_groups = {
    #     name: sorted(list(raw_names))
    #     for name, raw_names in final_filtered_model_groups.items()
    # }

    final_model_groups = {
        name: sorted(list(raw_names))
        for name, raw_names in final_filtered_model_groups.items()
        if len(raw_names) > 1
    }

    # Sort keys
    sorted_model_groups = dict(sorted(final_model_groups.items()))

    return sorted_model_groups

if __name__ == "__main__":
    pass
