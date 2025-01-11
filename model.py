import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForSequenceClassification

# Dictionary of models and paths
model_dict = {
    "GPT2": {"path": "./models/gpt2", "library": GPT2LMHeadModel, "tokenizer": GPT2Tokenizer, "use_pipeline": False},
    "GPT2-medium": {"path": "./models/gpt2-medium", "library": GPT2LMHeadModel, "tokenizer": GPT2Tokenizer, "use_pipeline": False},
    "GPT2-large": {"path": "./models/gpt2-large", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False},
    "GPT2-persian": {"path": "./models/gpt2-medium-persian", "library": GPT2LMHeadModel, "tokenizer": AutoTokenizer, "use_pipeline": False},
    "codegen": {"path": "./models/codegen", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False},
    "dialoGPT": {"path": "./models/dialogpt", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False},
    "dialoGPT-medium": {"path": "./models/dialogpt-medium", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False},
    "dialoGPT-large": {"path": "./models/dialogpt-large", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False},
    "GPT-Neo-125M": {"path": "./models/GPT-neo-125M", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": True},  # اضافه کردن مدل جدید
    "bert-emotion": {"path": "./models/bert-emotion", "library": AutoModelForSequenceClassification, "tokenizer": AutoTokenizer, "use_pipeline": True},
}

loaded_models = {}

def load_model_lazy(model_name):
    if not isinstance(model_name, str):
        raise ValueError(f"Model name must be a string, not {type(model_name)}")
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found!")

    model_info = model_dict[model_name]
    print(f"Loading model: {model_name}")

    # اگر مدل از pipeline پشتیبانی می‌کند
    if model_info.get("use_pipeline", False):
        print(f"Using pipeline for model: {model_name}")
        if model_name == "bert-emotion":
            # برای مدل bert-emotion از text-classification استفاده کنید
            model_pipeline = pipeline(
                "text-classification",  # تغییر وظیفه به text-classification
                model=model_info["path"],
                truncation=True
            )
        else:
            # برای سایر مدل‌ها از text-generation استفاده کنید
            model_pipeline = pipeline(
                "text-generation",
                model=model_info["path"],
                truncation=True,
                pad_token_id=50256
            )
        loaded_models[model_name] = {"pipeline": model_pipeline}
        return {"pipeline": model_pipeline}

    # در غیر این صورت، مدل و توکنایزر را به روش قدیمی بارگذاری کنید
    model = model_info["library"].from_pretrained(model_info["path"])
    tokenizer = model_info["tokenizer"].from_pretrained(model_info["path"])

    # تنظیمات پیش‌فرض
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}
    return {"model": model, "tokenizer": tokenizer}

def unload_model(model_name):
    global loaded_models
    if model_name in loaded_models:
        if "pipeline" in loaded_models[model_name]:
            del loaded_models[model_name]["pipeline"]
        elif "model" in loaded_models[model_name]:
            del loaded_models[model_name]["model"]
            del loaded_models[model_name]["tokenizer"]
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {model_name} unloaded and memory cleared.")
    else:
        print(f"Model {model_name} was not loaded.")