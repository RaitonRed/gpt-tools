import os
from importlib import import_module
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, model_info):
    model_class = model_info["library"]
    tokenizer_class = model_info["tokenizer"]
    
    save_dir = model_info["path"]
    hf_path = model_info.get("hf_path", model_info["path"])
    
    print(f"Downloading {model_name} from Hugging Face...")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        model = model_class.from_pretrained(hf_path)
        tokenizer = tokenizer_class.from_pretrained(hf_path)
        
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model {model_name} saved to {save_dir}")
        
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")
        raise