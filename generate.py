import torch

seed = 42


def generate_text(model_data, input_text, max_new_token, temperature=0.7, top_p=0.9, top_k=50):
    """
    Generate text using the given model and tokenizer with sampling (do_sample=True) and top-p (nucleus sampling).
    """
    if "pipeline" in model_data:
        model_pipeline = model_data["pipeline"]
        generated_text = model_pipeline(
            input_text,
            max_length=max_new_token,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,  # nucleus sampling
            top_k=top_k,
            truncation=True
        )[0]["generated_text"]
        return generated_text
    else:
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
            truncation=True,
            max_length=512
        )
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,  # nucleus sampling
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_code(model_data, prompt, max_new_tokens):
    """
    Generate code based on the provided prompt using sampling (do_sample=True) and top-p (nucleus sampling).
    """
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
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


def translate_text(input_text, model, tokenizer, max_length, mode):
    """
    Translate text between different languages using the provided model and tokenizer.
    """
    try:
        if mode == "English-French":
            prompt = f"translate English to French: {input_text}"
        elif mode == "French-English":
            prompt = f"translate French to English: {input_text}"
        elif mode == "Romanian-German":
            prompt = f"translate Romanian to German: {input_text}"
        elif mode == "German-Romanian":
            prompt = f"translate German to Romanian: {input_text}"
        elif mode == "English-German":
            prompt = f"translate English to German: {input_text}"
        else:
            return "Error: Invalid translation mode."

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        outputs = model.generate(input_ids, max_length=max_length)

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return f"Error during translation: {str(e)}"


def generate_poem(model_data, input_text, max_new_token, temperature=1.0, top_p=0.9, top_k=0):
    """
    Generate a poem using the provided model and tokenizer.
    """
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    input_ids = tokenizer.encode(input_text, return_tensors='tf')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_new_token,
        attention_mask=attention_mask,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.2,
        num_return_sequences=3
    )

    output = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

    return output