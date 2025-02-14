import streamlit as st
from database import create_db
from functions import *
from functions import _generate_code

# Supported models
models_options_general = ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-medium-persian', 'GPT-Neo-125M', 'GPT2-persian']
models_options_codegen = ['codegen-350M-mono', 'codegen-350M-multi']
models_options_chatbot = ['dialoGPT', 'dialoGPT-medium', 'dialoGPT-large', "Blenderbot-400M"]
models_options_summarization = ['Bart-large-CNN', 'bert-summary']
models_options_translation = ['T5-small']
translation_modes = ['English-French', 'French-English', 'Romanian-German', 'German-Romanian', 'English-German']

# Create database
create_db()

# Streamlit App
st.title("GPT Tools")
st.markdown("Generate something using GPT models. Select the model and adjust the parameters for optimal results.")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "Text Generator", "Multiverse Story Generator", "Interactive Story Writing", 
    "Training", "Code Generator", "Story World Builder", "Chatbot", "Text Summarization"
])

if option == "Text Generator":
    st.header("Text Generator")
    input_text = st.text_area("Input Text", placeholder="Enter your text here...", height=100)
    selected_model = st.radio("Select Model", models_options_general, index=0, key="text_generator_model")
    max_tokens = st.slider("Max New Tokens", 10, 900, 50, 1, key="max_new_tokens_slider_1")
    if st.button("Generate Text"):
        output_text = generate(input_text, selected_model, max_tokens)
        st.text_area("Generated Text", value=output_text, height=200, disabled=True)

elif option == "Multiverse Story Generator":
    st.header("Multiverse Story Generator")
    input_text = st.text_area("Enter your story idea", placeholder="e.g. A scientist discovers a parallel universe...", height=100)
    selected_model = st.radio("Select Model for Story Generation", models_options_general, index=0, key="multiverse_story_model")
    max_length = st.slider("Max Length", 50, 300, 150, 1, key="max_new_tokens_slider_2")
    if st.button("Generate Parallel Worlds"):
        output_text = generate_multiverse(input_text, selected_model, max_length)
        st.text_area("Generated Worlds", value=output_text, height=300, disabled=True)

elif option == "Interactive Story Writing":
    st.header("Interactive Story Writing")
    story_input = st.text_area("Add to Story", placeholder="Enter your part of the story...", height=100)
    story_model = st.radio("Select Model", models_options_general, index=0, key="interactive_story_model")
    story_max_length = st.slider("Max Length", 50, 300, 50, 1, key="max_new_tokens_slider_3")
    if st.button("Generate Next Part"):
        story_text = interactive_story(story_input, story_model, story_max_length)
        st.text_area("Story So Far", value=story_text, height=300, disabled=True)
    if st.button("Reset Story"):
        reset_story()
        st.text_area("Story So Far", value="", height=300, disabled=True)

elif option == "Training":
    st.header("Train Model")
    train_model_selector = st.radio("Select Model for Training", models_options_general, index=0)
    train_method = st.radio("Training Method", ["Custom Text", "Database", "Dataset File", "Hugging Face Dataset"], index=0)
    dataset_name = st.text_input("Hugging Face Dataset Name", placeholder="Enter dataset name (e.g., ag_news)")
    split_name = st.text_input("Dataset Split", placeholder="e.g., train, test, validation")
    epochs = st.slider("Epochs", 1, 100, 10, 1, key="epochs_slider")
    batch_size = st.slider("Batch Size", 1, 100, 8, 1, key="batch_size_slider")
    password = st.text_input("Enter Training Password", placeholder="Enter password", type="password")
    custom_text = st.text_area("Custom Text (optional)", placeholder="Enter custom text for training...", height=100)
    dataset_file = st.file_uploader("Upload Dataset", type=[".parquet", ".csv", ".json", ".txt"])
    if st.button("Train Model"):
        train_status = verify_and_train_combined(train_model_selector, train_method, epochs, batch_size, password, custom_text, dataset_file, dataset_name, split_name)
        st.text_area("Training Status", value=train_status, height=100, disabled=True)

elif option == "Code Generator":
    st.header("Code Generator")
    code_prompt = st.text_area("Code Prompt", placeholder="Describe your coding task, e.g., 'Write a Python function to calculate Fibonacci numbers.'", height=100)
    code_max_tokens = st.slider("Max Tokens", 10, 500, 150, 10, key="max_new_tokens_slider_4")
    selected_model = st.radio("Select model", models_options_codegen, index=0)
    if st.button("Generate Code"):
        generated_code = _generate_code(code_prompt, code_max_tokens, selected_model)
        st.text_area("Generated Code", value=generated_code, height=300, disabled=True)

elif option == "Story World Builder":
    st.header("Story World Builder")
    world_name = st.text_input("World Name", placeholder="Enter your world name...")
    locations = st.text_input("Locations", placeholder="Enter locations separated by commas...")
    characters = st.text_input("Characters", placeholder="Enter characters separated by commas...")
    if st.button("Create World"):
        world_status = define_world(world_name, locations, characters)
        st.text_area("World Status", value=world_status, height=100, disabled=True)
    
    st.markdown("### Generate a Story in Your World")
    story_world = st.text_input("Enter World Name", placeholder="World name...")
    event = st.text_input("Event", placeholder="Describe an event in the world...")
    selected_model = st.radio("Select Model", models_options_general, index=0)
    max_length = st.slider("Max Length", 50, 300, 150, 1, key="max_new_tokens_slider_5")
    if st.button("Generate Story"):
        generated_story = generate_story(selected_model, story_world, event, max_length)
        st.text_area("Generated Story", value=generated_story, height=300, disabled=True)

elif option == "Chatbot":
    st.header("Chatbot")
    username = st.text_input("Username", placeholder="Enter your username")
    chat_id = st.text_input("Chat ID (optional)", placeholder="Enter chat ID or leave blank for a new chat")
    selected_model = st.radio("Select Model", models_options_chatbot, index=0)
    input_text = st.text_area("Your Message", placeholder="Type your message here...", height=100)
    if st.button("Send"):
        chat_output, chat_id, emotion_output = chatbot_response_with_emotion(username, input_text, selected_model, chat_id)
        st.text_area("Chat History", value=chat_output, height=300, disabled=True)
        st.text_area("Detected Emotion", value=emotion_output, height=50, disabled=True)
    if st.button("Reset Chat"):
        reset_chat(username)
        st.text_area("Chat History", value="", height=300, disabled=True)
    
    st.markdown("### Fetch Chat IDs")
    username = st.text_input("Username", placeholder="Enter your username", key="fetch_username")
    if st.button("Fetch"):
        fetch_output = chat_ids(username)
        st.text_area("Chat IDs", value=fetch_output, height=100, disabled=True)

elif option == "Text Summarization":
    st.header("Text Summarization")
    text_input = st.text_area("Text input", placeholder="Enter your text here...", height=200)
    max_length = st.slider("Max Length", 50, 300, 130, 1, key="max_tokens_slider_6")
    min_length = st.slider("Min Length", 10, 100, 30, 1, key="min_tokens_slider_7")
    selected_model = st.radio("Select Model", models_options_summarization, index=0)
    if st.button("Summarize Text"):
        summary_output = handle_summarization(text_input, selected_model, max_length, min_length)
        st.text_area("Summary", value=summary_output, height=200, disabled=True)

st.markdown("Made by **AliDev.X** with ❤️")