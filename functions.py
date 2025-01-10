from model import load_model_lazy, unload_model
from generate import generate_code, generate_text
from database import *
import train

train_pass = '6818'

# AI-Powered Story World Builder Functions
world_data = {}

def _generate_code(code_prompt, max_tokens, selected_model='codegen'):
    """
    Generate code based on the code prompt and selected model.
    """
    # Load the model lazily
    model_data = load_model_lazy(selected_model)

    # Generate code
    generated_code = generate_code(model_data, code_prompt, max_tokens)

    # Unload the model after use
    unload_model(selected_model)

    return generated_code

def generate(input_text, selected_model, max_new_token, seed=None):
    """
    Generate text based on the selected model and input text.
    """
    # Load the model lazily
    model_data = load_model_lazy(selected_model)

    # Generate text
    generated_text = generate_text(model_data, input_text, max_new_token, seed)
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

def generate_multiverse(input_text, selected_model, max_new_tokens, num_worlds=3, seed=None):
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
        generated_text = generate(world_intro, selected_model, max_new_tokens, seed)

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
