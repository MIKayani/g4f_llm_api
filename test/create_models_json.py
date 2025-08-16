import g4f
import json
import re
from collections import defaultdict

def normalize_model_name(name):
    """
    Normalizes a model name to group similar models.
    - Converts to lowercase
    - Removes common prefixes like 'hf:', 'meta-llama/', etc.
    - Replaces underscores with hyphens
    - Inserts missing hyphens between letters and numbers
    - Removes '.0' from version numbers (e.g., 2.0 -> 2)
    - Removes various suffixes
    - Truncates after 'distill'
    - Removes date-like patterns (e.g., -06-17, 2024)
    - Removes leading/trailing numbers and hyphens from the name
    """
    s = name.lower()
    # Remove common prefixes like 'hf:', 'meta-llama/', etc.
    if ':' in s:
        s = s.split(':', 1)[-1]
    if '/' in s:
        s = s.split('/', 1)[-1]
    
    s = s.replace('_', '-')
    s = re.sub(r'([a-zA-Z])(\d)', r'\1-\2', s)
    s = re.sub(r'(\d+)\.0\b', r'\1', s)
    
    # remove date-like patterns (moved up)
    s = re.sub(r'-\d{2}-\d{2}', '', s) # e.g., -06-17
    s = re.sub(r'\b\d{4}\b', '', s)   # e.g., 2024, 0232
    s = re.sub(r'\b\d{6}', '', s)   # e.g., 202401
    s = re.sub(r'\b\d{8}', '', s)   # e.g., 20240101

    # Truncate after 'distill' (moved up)
    distill_match = re.search(r'distill', s)
    if distill_match:
        s = s[:distill_match.end()]

    # remove suffix (now without $)
    s = re.sub(r'-latest', '', s)
    s = re.sub(r'-exp', '', s)
    s = re.sub(r'-experimental', '', s)
    s = re.sub(r'-with-apps', '', s)
    s = re.sub(r'-preview', '', s)
    s = re.sub(r'-distill', '', s)
    s = re.sub(r'api', '', s) 
    s = re.sub(r'audio', '', s) 
    s = re.sub(r'chat', '', s) 
    s = re.sub(r'Experimental', '', s) 

    # remove leading and trailing numbers and hyphens
    s = re.sub(r'^["\d-]+|["\d-]+$', '', s)
    # remove extra spaces
    s = ' '.join(s.split())
    return s

def create_model_json():
    """
    Gathers all models from g4f providers, groups them by a normalized name,
    filters out smaller models, and saves the result to a JSON file.
    """
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
                        # if "audio" in model_name or "image" in model_name or "tts" in model_name or "coder" in model_name:
                        #     continue # Skip this model
                        # --- New filtering logic starts here ---
                        try:
                            normalized_name.encode('ascii')
                        except UnicodeEncodeError:
                            # If encoding to ascii fails, it means it contains non-ascii characters
                            continue # Skip this model
                        # --- New filtering logic ends here ---
                        model_groups[normalized_name].add(model_name)

    # Filter out smaller models
    filtered_model_groups = {}
    for name, raw_names in model_groups.items():
        keep_model = True
        # Find numbers followed by 'b' or 'm' (for billion/million parameters)
        matches = re.finditer(r'(\d+(?:\.\d+)?)([bm])', name)
        for match in matches:
            try:
                number = float(match.group(1))
                if number < 30:
                    # Check if the pattern is preceded by 'distilled'
                    preceding_text = name[:match.start()]
                    words_before = preceding_text.split()
                    if not (words_before and words_before[-1].endswith('distilled')):
                        keep_model = False
                        break
            except ValueError:
                # Ignore if the number part is not a valid float
                pass
        
        if keep_model:
            filtered_model_groups[name] = raw_names

    # Define famous LLM model names for filtering
    famous_llm_names = ["image", "diffusion", "flux", "kandinsky", "stable", "sdxl", "stablediffusion", 'imagen', 'xl', 'art', 'hd', 'anima', 'imagen']

    # Apply the new filtering based on famous LLM names
    final_filtered_model_groups = {}
    for name, raw_names in filtered_model_groups.items():
        for llm_name in famous_llm_names:
            if name.startswith(llm_name):
                final_filtered_model_groups[name] = raw_names
                break # Found a match, move to the next model

    # Convert sets to sorted lists for consistent JSON output
    # Use final_filtered_model_groups instead of filtered_model_groups
    final_model_groups = {
        name: sorted(list(raw_names))
        for name, raw_names in final_filtered_model_groups.items()
    }
    
    # Sort the top-level keys for a consistent output
    sorted_model_groups = dict(sorted(final_model_groups.items()))

    with open('models.json', 'w') as f:
        json.dump(sorted_model_groups, f, indent=2)

    print("Successfully created models.json with filtered and grouped model names.")

if __name__ == "__main__":
    create_model_json()