import argparse
import asyncio
from core import LLM, ImageGen
from core.llm.models import get_llm_models
from core.image.models import get_image_models

def main():
    parser = argparse.ArgumentParser(description='Chat with a free LLM or generate an image.')
    parser.add_argument('--list_models', action='store_true', help='List available models and exit.')
    parser.add_argument('--model_type', type=str, default='llm', choices=['llm', 'image'], help='The type of model to use.')
    parser.add_argument('--model_name', type=str, help='The name of the model to use.')
    parser.add_argument('--prompt', type=str, help='The prompt to send to the model.')
    args = parser.parse_args()

    if args.list_models:
        print("Available LLM models:")
        llm_models = get_llm_models()
        for model in llm_models.keys():
            print(f"  - {model}")

        print("\nAvailable Image models:")
        image_models = get_image_models()
        for model in image_models.keys():
            print(f"  - {model}")
        return

    if not args.model_name or not args.prompt:
        parser.error("--model_name and --prompt are required unless --list_models is specified.")

    if args.model_type == 'llm':
        model = LLM(model_name=args.model_name)
        response = model.chat(args.prompt)
    elif args.model_type == 'image':
        model = ImageGen(model_name=args.model_name)
        response = asyncio.run(model.generate(args.prompt))
    else:
        parser.error("Invalid model type specified.")

    if response:
        print(response)

if __name__ == '__main__':
    main()
