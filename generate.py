import torch

seed = 0

def generate_text(model_data, input_text, max_new_token):
    """
    Generate text using the given model and tokenizer.
    """
    if "pipeline" in model_data:
        # اگر مدل از pipeline پشتیبانی می‌کند
        model_pipeline = model_data["pipeline"]
        generated_text = model_pipeline(input_text, max_length=max_new_token, do_sample=True)[0]["generated_text"]
        return generated_text
    else:
        # روش قدیمی برای مدل‌هایی که از pipeline پشتیبانی نمی‌کنند
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_token,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_code(model_data, prompt, max_new_tokens):
    """
    Generate code based on the provided prompt using a code-specific model.
    """
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # تنظیم seed برای خروجی ثابت
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # توکنایز کردن ورودی
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # ایجاد attention mask
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)  # ایجاد یک ماسک توجه برای ورودی‌ها

    # تولید کد
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # ارسال attention mask
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,  # تنظیم شناسه توکن پایان به عنوان پرکننده
        repetition_penalty=1.2,  # جلوگیری از تکرار
        no_repeat_ngram_size=3,  # جلوگیری از تکرار n-gram
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)