import torch
import gc
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

# Dictionary of models and paths
model_dict = {
    "GPT2": {"path": "./models/gpt2", "library": GPT2LMHeadModel, "tokenizer": GPT2Tokenizer, "use_pipeline": False, "legacy": False},
    "GPT2-medium": {"path": "./models/gpt2-medium", "library": GPT2LMHeadModel, "tokenizer": GPT2Tokenizer, "use_pipeline": False, "legacy": False},
    "GPT2-large": {"path": "./models/gpt2-large", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "GPT2-medium-persian": {"path": "./models/gpt2-medium-persian", "library": GPT2LMHeadModel, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "codegen-350M-mono": {"path": "./models/codegen-350M-mono", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "codegen-350M-multi": {"path": "./models/codegen-350M-multi", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "dialoGPT": {"path": "./models/dialogpt", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "dialoGPT-medium": {"path": "./models/dialogpt-medium", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "dialoGPT-large": {"path": "./models/dialogpt-large", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "GPT-Neo-125M": {"path": "./models/GPT-neo-125M", "library": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "use_pipeline": True, "legacy": False},
    "bert-emotion": {"path": "./models/bert-emotion", "library": AutoModelForSequenceClassification, "tokenizer": AutoTokenizer, "use_pipeline": True, "legacy": False},
    "GPT2-persian": {"path": "./models/gpt2-persian", "library": GPT2LMHeadModel, "tokenizer": AutoTokenizer, "use_pipeline": True, "legacy": False},
    "Bart-large-CNN": {"path": "./models/bart-large", "library": AutoModelForSeq2SeqLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "bert-summary": {"path": "./models/bert-summary", "library": AutoModelForSeq2SeqLM, "tokenizer": AutoTokenizer, "use_pipeline": False, "legacy": False},
    "T5-small": {"path": "./models/t5-small", "library": T5ForConditionalGeneration, "tokenizer": T5Tokenizer, "use_pipeline": False, "legacy": False}
}

loaded_models = {}  # کش برای مدل‌های بارگذاری‌شده

def load_model_lazy(model_name):
    if not isinstance(model_name, str):
        raise ValueError(f"Model name must be a string, not {type(model_name)}")
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found!")

    # بررسی وجود مدل در کش
    if model_name in loaded_models:
        print(f"Model {model_name} already loaded. Using cached version.")
        loaded_models[model_name]["last_used"] = time.time()  # به‌روزرسانی زمان آخرین استفاده
        return loaded_models[model_name]["data"]

    # بارگذاری مدل جدید
    model_info = model_dict[model_name]
    print(f"Loading model: {model_name}")

    if model_info.get("use_pipeline", False):
        if model_name == "bert-emotion":
            model_pipeline = pipeline("text-classification", model=model_info["path"], truncation=True)
        else:
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

    # ذخیره مدل در کش
    loaded_models[model_name] = {"data": model_data, "last_used": time.time()}
    return model_data

def unload_model(model_name):
    global loaded_models
    if model_name in loaded_models:
        if "pipeline" in loaded_models[model_name]["data"]:
            del loaded_models[model_name]["data"]["pipeline"]
        elif "model" in loaded_models[model_name]["data"]:
            del loaded_models[model_name]["data"]["model"]
            del loaded_models[model_name]["data"]["tokenizer"]
        del loaded_models[model_name]
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {model_name} unloaded and memory cleared.")
    else:
        print(f"Model {model_name} was not loaded.")

def unload_inactive_models(max_inactive_time=300):  # 300 ثانیه = ۵ دقیقه
    current_time = time.time()
    inactive_models = [model_name for model_name, model_info in loaded_models.items() if current_time - model_info["last_used"] > max_inactive_time]
    for model_name in inactive_models:
        unload_model(model_name)

# زمان‌بندی تخلیه مدل‌های غیرفعال
def schedule_unload_inactive_models(interval=300):  # هر ۵ دقیقه اجرا شود
    def task():
        unload_inactive_models()
        threading.Timer(interval, task).start()
    task()

# شروع زمان‌بندی
schedule_unload_inactive_models()