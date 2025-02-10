import torch
from model import load_model_lazy, unload_model
from generate import *
from database import *
import train
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

train_pass = '6818'

# AI-Powered Story World Builder Functions
world_data = {}

def _generate_code(code_prompt, max_tokens, selected_model):
    """
    Generate code based on the code prompt and selected model.
    """
    # Load the model lazily
    model_data = load_model_lazy(selected_model)

    # Generate code
    generated_code = generate_code(model_data, code_prompt, max_tokens)

    # Unload the model after use
    unload_model(selected_model)  # تخلیه مدل پس از استفاده

    return generated_code

def generate(input_text, selected_model, max_new_token):
    """
    Generate text based on the selected model and input text.
    """
    # Load the model lazily
    model_data = load_model_lazy(selected_model)

    # Generate text
    generated_text = generate_text(model_data, input_text, max_new_token)
    insert_into_db(input_text, selected_model)

    # Unload the model after use
    unload_model(selected_model)

    return generated_text

def define_world(world_name, locations, characters):
    """
    Define a new story world with locations and characters.
    """
    world_data["world_name"] = world_name
    world_data["locations"] = locations.split(", ")
    world_data["characters"] = characters.split(", ")
    return f"World '{world_name}' created with locations: {locations} and characters: {characters}"

def generate_story(model, world_name, event, max_length):
    """
    Generate a story based on the defined world and an event.
    """
    if not world_name or not world_data.get("world_name"):
        return "Error: Please define a world first."

    if world_name != world_data["world_name"]:
        return f"Error: World '{world_name}' not found. Define it first."

    prompt = f"In the world of {world_name}, {event}. Locations: {', '.join(world_data['locations'])}. Characters: {', '.join(world_data['characters'])}."

    generated_story = generate(prompt, model, max_length)
    return generated_story


# Story Mode
story = []

# Main Function For Story Generating
def interactive_story(input_text, selected_model, max_length):
    global story
    if input_text.strip():
        story.append(input_text)  # Add user input to story
    current_text = " ".join(story)  # Build cumulative story

    generated_text = generate(current_text, selected_model, max_length)
    story.append(generated_text)  # Add generated text to story

    return current_text + "\n\n" + generated_text


def reset_story():
    global story
    story = []  # Reset story
    return ""

def generate_multiverse(input_text, selected_model, max_new_tokens, num_worlds=3):
    """
    Generate multiple parallel worlds from a single input text.
    """
    worlds = []

    for i in range(num_worlds):
        world_intro = f"World {i + 1}: "
        # Custom logic for different parallel worlds
        if i == 0:
            world_intro += f"{input_text} This world leads to a parallel universe!"
        elif i == 1:
            world_intro += f"{input_text} In this world, time splits into different periods!"
        elif i == 2:
            world_intro += f"{input_text} This world faces a strange physical anomaly that changes everything!"

        # Generate the story for this world
        generated_text = generate(world_intro, selected_model, max_new_tokens)

        worlds.append(generated_text)

    return "\n\n".join(worlds)



# Function to verify password, train the model, and clear the database
def verify_and_train_combined(selected_model, train_method, epochs, batch_size, password, custom_text, dataset_file, dataset_name, split_name):
    if password != train_pass:
        return "Error: Incorrect password. Training not started."

    if train_method == "Custom Text" and custom_text.strip():
        train.train_model_with_text(selected_model, custom_text, epochs, batch_size)
        return f"Training completed for model: {selected_model} using custom text."

    elif train_method == "Database":
        train.train_model_with_database(selected_model, epochs, batch_size)
        clear_database()
        return f"Training completed for model: {selected_model} using database. Database cleared."

    elif train_method == "Dataset File" and dataset_file is not None:
        try:
            dataset_path = dataset_file.name
            train.train_model_with_dataset(selected_model, epochs, batch_size, dataset_path)
            return f"Training completed for model: {selected_model} using uploaded dataset."
        except Exception as e:
            return f"Error during training with dataset: {str(e)}"

    elif train_method == "Hugging Face Dataset" and dataset_name.strip():
        try:
            train.train_model_with_hf_dataset(selected_model, epochs, batch_size, dataset_name, split=split_name.strip())
            return f"Training completed for model: {selected_model} using Hugging Face dataset {dataset_name}."
        except Exception as e:
            return f"Error during training with Hugging Face dataset: {str(e)}"

    else:
        return "Error: Invalid input for training. Please check your selections."

def limit_chat_history(chat_history, max_turns=3):
    """
    محدود کردن تعداد پیام‌های تاریخچه به max_turns.
    """
    turns = chat_history.split("\n")
    if len(turns) > max_turns * 2:  # هر سوال و پاسخ دو خط می‌شود
        turns = turns[-max_turns * 2:]  # فقط n پیام اخیر را نگه می‌دارد
    return "\n".join(turns)

def chatbot_response(username, input_text, selected_model, chat_id=None):
    if not username.strip():
        return "Error: Please enter a username.", "", str(uuid.uuid4())  # تولید شناسه جدید

    # اگر شناسه چت وارد نشده باشد، یک شناسه جدید تولید می‌شود
    if not chat_id or chat_id.strip() == "":
        chat_id = str(uuid.uuid4())  # تولید شناسه جدید

    # Load model lazily
    model_data = load_model_lazy(selected_model)

    # Retrieve previous chats from database
    previous_chats = fetch_chats_by_id(chat_id)
    chat_history = "\n".join([f"User: {msg}\nAI: {resp}" for msg, resp in previous_chats])

    # محدود کردن تاریخچه چت
    if chat_history:
        chat_history = limit_chat_history(chat_history, max_turns=3)
        prompt = f"{chat_history}\nUser: {input_text}\nAI:"
    else:
        prompt = f"User: {input_text}\nAI:"

    # Generate response
    max_new_token = 150  # تعداد توکن‌های جدید
    full_response = generate_text(model_data, prompt, max_new_token)  # حذف آرگومان‌های اضافی

    # Extract only the new AI response
    ai_response = full_response.split("AI:")[-1].strip()

    unload_model(selected_model)

    # Save chat to database
    insert_chat(chat_id, username, input_text, ai_response)

    # Return updated chat history and chat_id
    updated_history = chat_history + f"\nUser: {input_text}\nAI: {ai_response}"
    return updated_history, chat_id

def chat_ids(username):
    return fetch_ids_by_user(username)

def reset_chat(username):
    clear_chats_by_username(username)  # حذف چت‌های مرتبط با کاربر
    return f"Chat history cleared for user: {username}", ""

# توابع تحلیل احساسات
def analyze_emotion(user_input):
    # بارگذاری مدل احساسات
    model_data = load_model_lazy("bert-emotion")
    
    # اگر مدل از pipeline پشتیبانی می‌کند
    if "pipeline" in model_data:
        emotion_pipeline = model_data["pipeline"]
        result = emotion_pipeline(user_input)
        emotion = result[0]['label']
        confidence = result[0]['score']
    else:
        # روش قدیمی برای مدل‌هایی که از pipeline پشتیبانی نمی‌کنند
        emotion_tokenizer = model_data['tokenizer']
        emotion_model = model_data['model']
        inputs = emotion_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = emotion_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        emotion = probs.argmax().item()
        confidence = probs.max().item()
    
    unload_model("bert-emotion")
    return emotion, confidence

def emotion_label(index):
    emotions = ["anger", "joy", "sadness", "fear", "love", "surprise"]
    return emotions[index]

def chatbot_response_with_emotion(username, input_text, selected_model, chat_id=None):
    if not username.strip():
        return "Error: Please enter a username.", "", str(uuid.uuid4())

    if not chat_id or chat_id.strip() == "":
        chat_id = str(uuid.uuid4())

    # بارگذاری مدل چت و احساسات
    model_data = load_model_lazy(selected_model)

    # تحلیل احساسات پیام کاربر
    emotion, confidence = analyze_emotion(input_text)
    user_emotion = emotion  # برچسب احساسات

    # بازیابی چت‌های قبلی از پایگاه داده
    previous_chats = fetch_chats_by_id(chat_id)
    chat_history = "\n".join([f"User: {msg}\nAI: {resp}" for msg, resp in previous_chats])

    # محدود کردن تاریخچه چت
    if chat_history:
        chat_history = limit_chat_history(chat_history, max_turns=3)
        prompt = f"[Emotion: {user_emotion}]\n{chat_history}\nUser: {input_text}\nAI:"
    else:
        prompt = f"[Emotion: {user_emotion}]\nUser: {input_text}\nAI:"

    # تولید پاسخ
    max_new_token = 150
    full_response = generate_text(model_data, prompt, max_new_token)

    # استخراج پاسخ AI
    ai_response = full_response.split("AI:")[-1].strip()

    # آزادسازی مدل‌ها
    unload_model(selected_model)
    unload_model("bert-emotion")

    # ذخیره چت در پایگاه داده
    insert_chat(chat_id, username, input_text, ai_response)

    # بازگرداندن تاریخچه به‌روز شده و شناسه چت
    updated_history = chat_history + f"\nUser: {input_text}\nAI: {ai_response}"
    return updated_history, chat_id, user_emotion


def handle_summarization(input_text, selected_model, max_length=130, min_length=30):
    """
    Handle the summarization process, including model loading and unloading.
    """
    # Load the model lazily
    model_data = load_model_lazy(selected_model)

    if model_data['model'] is None or model_data['tokenizer'] is None:
        print("Error: Model or tokenizer not loaded correctly!")
        return "Error: Model or tokenizer not loaded correctly!"
    else:
        print("Model and tokenizer loaded successfully!")
    
    try:
        if "pipeline" in model_data:
            # If the model supports pipeline
            summarizer = model_data["pipeline"]
            summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)

            if summary and isinstance(summary, list) and len(summary) > 0:
                if 'generated_text' in summary[0]:
                    result = summary[0]['generated_text']
                else:
                    print("'generated_text' key not found in the summarizer output")
                    result = f"Unexpected summarizer output format: {summary[0]}"
            else:
                print(f"Unexpected summarizer output: {summary}")
                result = "No summary generated. The summarizer returned an unexpected output."
        else:
            # If the model doesn't support pipeline, use the model and tokenizer directly
            result = summarize_text(input_text, model_data['model'], model_data['tokenizer'], max_length, min_length)

        
        return result
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return f"Error during summarization: {str(e)}"
    finally:
        unload_model(selected_model)

def handle_translation(input_text, selected_model, mode, max_length):
    model_data = load_model_lazy(selected_model)

    if model_data['model'] is None or model_data['tokenizer'] is None:
        print("Error: Model or tokenizer not loaded correctly!")
        return "Error: Model or tokenizer not loaded correctly!"
    else:
        print("Model and tokenizer loaded successfully!")

    try:
        result = translate_text(input_text, model_data['model'], model_data['tokenizer'], max_length, mode)
        return result
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return f"Error during translation: {str(e)}"
    finally:
        unload_model(selected_model)