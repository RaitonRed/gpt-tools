import torch
import gc
import os
import time
import threading
from download import download_model
from cachetools import cached, LRUCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, TFGPT2LMHeadModel

# Dictionary of models and their paths
model_dict = {
    "GPT2": {
        "path": "./models/gpt2", 
        "hf_path": "openai-community/gpt2",
        "library": GPT2LMHeadModel, 
        "tokenizer": GPT2Tokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "GPT2-medium": {
        "path": "./models/gpt2-medium", 
        "hf_path": "openai-community/gpt2-medium",
        "library": GPT2LMHeadModel, 
        "tokenizer": GPT2Tokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "GPT2-large": {
        "path": "./models/gpt2-large", 
        "hf_path": "openai-community/gpt2-large",
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "GPT2-poet": {
        "path": "./models/gpt2-poet", 
        "hf_path": "ashiqabdulkhader/GPT2-Poet",
        "library": TFGPT2LMHeadModel, 
        "tokenizer": GPT2Tokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "GPT2-medium-persian": {
        "path": "./models/gpt2-medium-persian", 
        "hf_path": "flax-community/gpt2-medium-persian",
        "library": GPT2LMHeadModel, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "codegen-350M-mono": {
        "path": "./models/codegen-350M-mono", 
        "hf_path": "Salesforce/codegen-350M-mono",
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "codegen-350M-multi": {
        "path": "./models/codegen-350M-multi", 
        "hf_path": "Salesforce/codegen-350M-multi",
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "dialoGPT": {
        "path": "./models/dialogpt", 
        "hf_path": "microsoft/DialoGPT-small",
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "dialoGPT-medium": {
        "path": "./models/dialogpt-medium",
        "hf_path": "microsoft/DialoGPT-medium",
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "dialoGPT-large": {
        "path": "./models/dialogpt-large", 
        "hf_path": "microsoft/DialoGPT-large", 
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "GPT-Neo-125M": {
        "path": "./models/GPT-neo-125M", 
        "hf_path": "EleutherAI/gpt-neo-125m", 
        "library": AutoModelForCausalLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": True, 
        "legacy": False
        },
    
    "bert-emotion": {
        "path": "./models/bert-emotion", 
        "hf_path": "", 
        "library": AutoModelForSequenceClassification, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": True, 
        "legacy": False
        },
    
    "GPT2-persian": {
        "path": "./models/gpt2-persian", 
        "hf_path": "bolbolzaban/gpt2-persian", 
        "library": GPT2LMHeadModel, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": True, 
        "legacy": False
        },
    
    "Bart-large-CNN": {
        "path": "./models/bart-large", 
        "hf_path": "facebook/bart-large-cnn",
        "library": AutoModelForSeq2SeqLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
        
    "bert-summary": {
        "path": "./models/bert-summary", 
        "hf_path": "Shobhank-iiitdwd/BERT_summary",
        "library": AutoModelForSeq2SeqLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "T5-small": {
        "path": "./models/t5-small", 
        "hf_path": "google-t5/t5-small",
        "library": T5ForConditionalGeneration, 
        "tokenizer": T5Tokenizer, 
        "use_pipeline": False, 
        "legacy": False
        },
    
    "Blenderbot-400M": {
        "path": "./models/blenderbot-400M", 
        "hf_path": "facebook/blenderbot-400M-distill",
        "library": AutoModelForSeq2SeqLM, 
        "tokenizer": AutoTokenizer, 
        "use_pipeline": False, 
        "legacy": False
        }
}

# Create an LRU cache for models with a maximum size of 3
model_cache = LRUCache(maxsize=3)

@cached(model_cache)
def load_model_lazy(model_name):
    """
    Load a model lazily (only when needed) and cache it using cachetools.
    """
    if not isinstance(model_name, str):
        raise ValueError(f"Model name must be a string, not {type(model_name)}")
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found!")
    
    if not os.path.exists(model_info["path"]):
        try:
            download_model(model_name, model_info)  # فراخوانی تابع دانلود
        except Exception as e:
            raise RuntimeError(f"Auto-download failed: {str(e)}")

    model_info = model_dict[model_name]
    print(f"Loading model: {model_name}")

    if model_info.get("use_pipeline", False):
        if model_name == "bert-emotion":
            model_pipeline = pipeline("text-classification", model=model_info["path"], truncation=True)
        else:
            print(f"Using pipeline for model {model_name}")
            model_pipeline = pipeline(
                "text-generation",
                model=model_info["path"],
                truncation=True,
                pad_token_id=50256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        model_data = {"pipeline": model_pipeline}
    else:
        model = model_info["library"].from_pretrained(model_info["path"])
        tokenizer = model_info["tokenizer"].from_pretrained(model_info["path"], legacy=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_data = {"model": model, "tokenizer": tokenizer}

    return model_data

def unload_model(model_name):
    """
    Unload a model from memory and clear GPU cache.
    """
    if model_name in model_cache:
        del model_cache[model_name]
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {model_name} unloaded and memory cleared.")
    else:
        print(f"Model {model_name} was not loaded.")

def unload_inactive_models(max_inactive_time=500):
    """
    Unload models that have been inactive for a specified time.
    """
    current_time = time.time()
    inactive_models = [model_name for model_name, model_info in model_cache.items() if current_time - model_info["last_used"] > max_inactive_time]
    for model_name in inactive_models:
        unload_model(model_name)

def schedule_unload_inactive_models(interval=500):
    """
    Schedule a periodic task to unload inactive models.
    """
    def task():
        unload_inactive_models()
        threading.Timer(interval, task).start()
    task()