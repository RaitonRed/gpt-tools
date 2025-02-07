import torch

seed = 42

def generate_text(model_data, input_text, max_new_token, temperature=0.7, top_p=0.9, top_k=50):
    """
    Generate text using the given model and tokenizer with sampling (do_sample=True) and top-p (nucleus sampling).
    """
    if "pipeline" in model_data:
        # اگر مدل از pipeline پشتیبانی می‌کند
        model_pipeline = model_data["pipeline"]
        generated_text = model_pipeline(
            input_text,
            max_length=max_new_token,
            do_sample=True,  # فعال کردن نمونه‌گیری تصادفی
            temperature=temperature,  # کنترل میزان خلاقیت
            top_p=top_p, # نمونه‌گیری هسته‌ای (nucleus sampling)
            top_k=top_k,
            truncation=True  # فعال کردن truncation
        )[0]["generated_text"]
        return generated_text
    else:
        # روش قدیمی برای مدل‌هایی که از pipeline پشتیبانی نمی‌کنند
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        encodings = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,  # فعال کردن truncation
            max_length=512
        )
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_token,
            do_sample=True,  # فعال کردن نمونه‌گیری تصادفی
            temperature=temperature,  # کنترل میزان خلاقیت
            top_p=top_p,  # نمونه‌گیری هسته‌ای (nucleus sampling)
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # جلوگیری از تکرار
            no_repeat_ngram_size=3,  # جلوگیری از تکرار n-gram
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_code(model_data, prompt, max_new_tokens):
    """
    Generate code based on the provided prompt using sampling (do_sample=True) and top-p (nucleus sampling).
    """
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # تنظیم seed برای خروجی ثابت
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # توکنایز کردن ورودی
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # ایجاد attention mask
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    # تولید کد
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # فعال کردن نمونه‌گیری تصادفی
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # جلوگیری از تکرار
        no_repeat_ngram_size=3,  # جلوگیری از تکرار n-gram
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(input_text, model, tokenizer, max_length=130, min_length=30):
    """
    Summarize the input text using the provided model and tokenizer.
    """
    try:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )

        result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return result
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return f"Error during summarization: {str(e)}"
