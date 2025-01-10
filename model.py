import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Dictionary of models and paths
model_dict = {
    "gpt2": {"path": "./models/gpt2", "library": GPT2LMHeadModel, "tokenizer": GPT2Tokenizer},
    "gpt2-medium": {"path": "./models/gpt2-medium", "library": GPT2LMHeadModel, "tokenizer": GPT2Tokenizer},
    "gpt2-large": {"path": "./models/gpt2-large", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer},
    "gpt2-persian": {"path": "./models/gpt2-medium-persian", "library": GPT2LMHeadModel, "tokenizer": AutoTokenizer},
    "codegen": {"path": "./models/codegen", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer},
}

loaded_models = {}

def load_model_lazy(model_name):
    """
    Lazy loading of the model
    """
    if not isinstance(model_name, str):
        raise ValueError(f"Model name must be a string, not {type(model_name)}")
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found!")

    model_info = model_dict[model_name]
    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    model = model_info["library"].from_pretrained(model_info["path"])
    tokenizer = model_info["tokenizer"].from_pretrained(model_info["path"])

    # Default settings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}

    return {"model": model, "tokenizer": tokenizer}


def unload_model(model_name):
    global loaded_models
    if model_name in loaded_models:
        del loaded_models[model_name]
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {model_name} unloaded and memory cleared.")
    else:
        print(f"Model {model_name} was not loaded.")
