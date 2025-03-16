import sys
from generate import *
from database import *
from model import load_model_lazy
import uuid
import logging
import uvicorn
from api import app
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from ollama import chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor()

# AI-Powered Story World Builder Functions
world_data = {}

def _generate_code(code_prompt, max_tokens, selected_model):
    model_data = load_model_lazy(selected_model)
    generated_code = generate_code(model_data, code_prompt, max_tokens)
    return generated_code

async def generate(input_text, selected_model, max_new_token):
    if not input_text.strip():
        return "Error: Input text cannot be empty."

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    model_data = await loop.run_in_executor(executor, load_model_lazy, selected_model)
    generated_text = await loop.run_in_executor(executor, generate_text, model_data, input_text, max_new_token)
    await loop.run_in_executor(executor, insert_into_db, input_text, selected_model)
    return generated_text

def define_world(world_name, locations, characters):
    world_data["world_name"] = world_name
    world_data["locations"] = locations.split(", ")
    world_data["characters"] = characters.split(", ")
    return f"World '{world_name}' created with locations: {locations} and characters: {characters}"

async def generate_story(model, world_name, event, max_length):
    if not world_name or not world_data.get("world_name"):
        return "Error: Please define a world first."
    if world_name != world_data["world_name"]:
        return f"Error: World '{world_name}' not found. Define it first."
    prompt = f"In the world of {world_name}, {event}. Locations: {', '.join(world_data['locations'])}. Characters: {', '.join(world_data['characters'])}."
    return await asyncio.run(generate(prompt, model, max_length))

# Story Mode
story = []

async def interactive_story(input_text, selected_model, max_length):
    global story
    if input_text.strip():
        story.append(input_text)
    current_text = " ".join(story)
    generated_text = await asyncio.run(generate(current_text, selected_model, max_length))
    story.append(generated_text)
    return current_text + "\n\n" + generated_text

def reset_story():
    global story
    story = []
    return ""

async def generate_multiverse(input_text, selected_model, max_new_tokens, num_worlds=3):
    worlds = []
    for i in range(num_worlds):
        world_intro = f"World {i + 1}: {input_text} "
        if i == 0:
            world_intro += "This world leads to a parallel universe!"
        elif i == 1:
            world_intro += "In this world, time splits into different periods!"
        elif i == 2:
            world_intro += "This world faces a strange physical anomaly that changes everything!"
        worlds.append(await asyncio.run(generate(world_intro, selected_model, max_new_tokens)))
    return "\n\n".join(worlds)

def limit_chat_history(chat_history, max_turns=6):
    turns = chat_history.split("\n")
    if len(turns) > max_turns * 2:
        turns = turns[-max_turns * 2:]
    return "\n".join(turns)

def chatbot_response(username, input_text, selected_model, chat_id=None):
    if not username.strip():
        return "Error: Please enter a username.", "", str(uuid.uuid4())
    if not chat_id or chat_id.strip() == "":
        chat_id = str(uuid.uuid4())
    previous_chats = fetch_chats_by_id(chat_id)
    chat_history = "\n".join([f"User: {msg}\nAI: {resp}" for msg, resp in previous_chats])
    if chat_history:
        chat_history = limit_chat_history(chat_history, max_turns=6)
        prompt = f"{chat_history}\nUser: {input_text}\nAI:"
    else:
        prompt = f"User: {input_text}\nAI:"
    max_new_token = 250
    full_response = asyncio.run(generate(prompt, selected_model, max_new_token))
    ai_response = full_response.split("AI:")[-1].strip()
    insert_chat(chat_id, username, input_text, ai_response)
    updated_history = chat_history + f"\nUser: {input_text}\nAI: {ai_response}"
    return updated_history, chat_id

def chat_ids(username):
    return fetch_ids_by_user(username)

def reset_chat(username):
    clear_chats_by_username(username)
    return f"Chat history cleared for user: {username}", ""

def analyze_emotion(user_input):
    model_data = load_model_lazy("bert-emotion")
    if "pipeline" in model_data:
        emotion_pipeline = model_data["pipeline"]
        result = emotion_pipeline(user_input)
        emotion = result[0]['label']
        confidence = result[0]['score']
    else:
        emotion_tokenizer = model_data['tokenizer']
        emotion_model = model_data['model']
        inputs = emotion_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = emotion_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        emotion = probs.argmax().item()
        confidence = probs.max().item()
    return emotion, confidence

def emotion_label(index):
    emotions = ["anger", "joy", "sadness", "fear", "love", "surprise"]
    return emotions[index]

def chatbot_response_with_emotion(username, input_text, selected_model, chat_id=None):
    if not username.strip():
        return "Error: Please enter a username.", "", str(uuid.uuid4())
    if not chat_id or chat_id.strip() == "":
        chat_id = str(uuid.uuid4())
    model_data = load_model_lazy(selected_model)
    emotion, confidence = analyze_emotion(input_text)
    user_emotion = emotion
    previous_chats = fetch_chats_by_id(chat_id)
    chat_history = "\n".join([f"User: {msg}\nAI: {resp}" for msg, resp in previous_chats])
    if chat_history:
        chat_history = limit_chat_history(chat_history, max_turns=6)
        prompt = f"[Emotion: {user_emotion}]\n{chat_history}\nUser: {input_text}\nAI:"
    else:
        prompt = f"[Emotion: {user_emotion}]\nUser: {input_text}\nAI:"
    max_new_token = 250
    full_response = generate_text(model_data, prompt, max_new_token)
    ai_response = full_response.split("AI:")[-1].strip()
    insert_chat(chat_id, username, input_text, ai_response)
    updated_history = chat_history + f"\nUser: {input_text}\nAI: {ai_response}"
    return updated_history, chat_id

async def handle_summarization(input_text, selected_model, max_length=130, min_length=30):
    model_data = load_model_lazy(selected_model)
    if model_data['model'] is None or model_data['tokenizer'] is None:
        return "Error: Model or tokenizer not loaded correctly!"
    try:
        if "pipeline" in model_data:
            summarizer = model_data["pipeline"]
            summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
            if summary and isinstance(summary, list) and len(summary) > 0:
                if 'generated_text' in summary[0]:
                    result = summary[0]['generated_text']
                else:
                    result = f"Unexpected summarizer output format: {summary[0]}"
            else:
                result = "No summary generated. The summarizer returned an unexpected output."
        else:
            result = summarize_text(input_text, model_data['model'], model_data['tokenizer'], max_length, min_length)
        return result
    except Exception as e:
        return f"Error during summarization: {str(e)}"

async def handle_translation(input_text, selected_model, mode, max_length):
    model_data = load_model_lazy(selected_model)
    if model_data['model'] is None or model_data['tokenizer'] is None:
        return "Error: Model or tokenizer not loaded correctly!"
    try:
        result = translate_text(input_text, model_data['model'], model_data['tokenizer'], max_length, mode)
        return result
    except Exception as e:
        return f"Error during translation: {str(e)}"

def run_fastapi():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

async def generate_entertainment_content(topic, content_type, model, max_length=500):
    if content_type == "joke":
        prompt = f"Generate a funny joke about: {topic}"
    elif content_type == "story":
        prompt = f"Generate a short story about: {topic}"
    elif content_type == "riddle":
        prompt = f"Generate a riddle about: {topic}"
    elif content_type == "poem":
        prompt = f"Generate a poem about: {topic}"
    else:
        return "Invalid content type"
    return await generate(prompt, model, max_length)

def signal_handler(_, __):
    print("\nKeyboard Interruption. Shutting down application")
    sys.exit(0)
    
async def handle_deepseek_message(prompt, selected_model):
    """
    Generate a response using the deepseek-r1:1.5b model with streaming.
    """
    if selected_model == "DeepSeek R1 (1.5B)":
        model = "deepseek-r1:1.5b"
    else:
        return "Model not found!"

    try:
        stream = chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True)
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            yield response  # Stream the response incrementally
    except Exception as e:
        yield f"Error: {str(e)}"