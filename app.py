import gradio as gr
from database import create_db
from functions import *
from functions import _generate_code

# Supported models
models_options_general = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-persian']
models_options_codegen = ['codegen']

# Create database
create_db()

# Interface setup
with gr.Blocks() as interface:
    gr.Markdown(
        "# **GPT Tools**\n\n"
        "Generate something using GPT models. Select the model and adjust the parameters for optimal results."
    )
    with gr.Tabs():
        with gr.Tab("Text Generator"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    input_text = gr.Textbox(label="Input Text", placeholder="Enter your text here...", lines=4, max_lines=6)
                    selected_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model", type="value")
                    with gr.Row():
                        max_tokens = gr.Slider(10, 100, value=50, step=1, label="Max New Tokens", interactive=True)
                with gr.Column(scale=1, min_width=350):
                    output_text = gr.Textbox(label="Generated Text", interactive=False, lines=8, max_lines=12)
                    generate_button = gr.Button("Generate Text", variant="primary")

            generate_button.click(
                generate,
                inputs=[input_text, selected_model, max_tokens],
                outputs=output_text,
            )


        with gr.Tab("Multiverse Story Generator"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    input_text = gr.Textbox(label="Enter your story idea", placeholder="e.g. A scientist discovers a parallel universe...", lines=4, max_lines=6)
                    selected_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model for Story Generation", type="value")
                    max_length = gr.Slider(50, 300, value=150, step=1, label="Max Length", interactive=True)

                with gr.Column(scale=1, min_width=350):
                    output_text = gr.Textbox(label="Generated Worlds", interactive=False, lines=12, max_lines=20)
                    generate_button = gr.Button("Generate Parallel Worlds", variant="primary")

            generate_button.click(
                generate_multiverse,
                inputs=[input_text, selected_model, max_length],
                outputs=output_text,
            )

        with gr.Tab("Interactive Story Writing"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    story_input = gr.Textbox(label="Add to Story", placeholder="Enter your part of the story...", lines=4, max_lines=6)
                    story_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model", type="value")
                    story_max_length = gr.Slider(50, 300, value=50, step=1, label="Max Length", interactive=True)
                with gr.Column(scale=1, min_width=350):
                    story_text = gr.Textbox(label="Story So Far", interactive=False, lines=12, max_lines=20)
                    story_button = gr.Button("Generate Next Part", variant="primary")
                    reset_button = gr.Button("Reset Story", variant="secondary")

            story_button.click(
                interactive_story,
                inputs=[story_input, story_model, story_max_length],
                outputs=story_text,
            )
            reset_button.click(
                reset_story,
                inputs=[],
                outputs=story_text,
            )

        with gr.Tab("Training"):
            gr.Markdown("# **Train Model**\n\n")
            with gr.Column(scale=1, min_width=250):
                train_model_selector = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model for Training", type="value")
                train_method = gr.Radio(
                    choices=["Custom Text", "Database", "Dataset File", "Hugging Face Dataset"],
                    value="Custom Text",
                    label="Training Method",
                    type="value"
                )
                dataset_name = gr.Textbox(label="Hugging Face Dataset Name", placeholder="Enter dataset name (e.g., ag_news)")
                split_name = gr.Textbox(label="Dataset Split", placeholder="e.g., train, test, validation")
                epochs = gr.Slider(1, 100, value=10, step=1, label="Epochs", interactive=True)
                batch_size = gr.Slider(1, 100, value=8, step=1, label="Batch Size", interactive=True)
                password = gr.Textbox(label="Enter Training Password", placeholder="Enter password", type="password")
                custom_text = gr.Textbox(label="Custom Text (optional)", placeholder="Enter custom text for training...")
                dataset_file = gr.File(label="Upload Dataset", type="filepath", file_types=[".parquet", ".csv", ".json", ".txt"])
                train_button = gr.Button("Train Model", variant="primary")
                train_status = gr.Textbox(label="Training Status", interactive=False)
            
            train_button.click(
                verify_and_train_combined,
                inputs=[train_model_selector, train_method, epochs, batch_size, password, custom_text, dataset_file, dataset_name, split_name],
                outputs=train_status,
            )
            train_button.click(
                verify_and_train_combined,
                inputs=[train_model_selector, train_method, epochs, batch_size, password, custom_text, dataset_file, dataset_name, split_name],
                outputs=train_status,
            )

        with gr.Tab("Code Generator"):
            gr.Markdown("### Generate Code from Descriptions")
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    code_prompt = gr.Textbox(label="Code Prompt", placeholder="Describe your coding task, e.g., 'Write a Python function to calculate Fibonacci numbers.'")
                    code_max_tokens = gr.Slider(10, 500, value=150, step=10, label="Max Tokens")
                with gr.Column(scale=1, min_width=350):
                    generated_code = gr.Textbox(label="Generated Code", interactive=False, lines=10, max_lines=20)
                    generate_code_button = gr.Button("Generate Code")

            generate_code_button.click(
                _generate_code,
                inputs=[code_prompt, code_max_tokens],
                outputs=generated_code,
            )

        # Add AI-Powered Story World Builder Tab
        with gr.Tab("Story World Builder"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    world_name = gr.Textbox(label="World Name", placeholder="Enter your world name...")
                    locations = gr.Textbox(label="Locations", placeholder="Enter locations separated by commas...")
                    characters = gr.Textbox(label="Characters", placeholder="Enter characters separated by commas...")
                    create_button = gr.Button("Create World", variant='primary')
                    generate_story_button = gr.Button("Generate Story")
                with gr.Column(scale=1, min_width=350):
                    world_status = gr.Textbox(label="World Status", interactive=False)
                    generated_story = gr.Textbox(label="Generated Story", interactive=False, lines=12, max_lines=20)


            create_button.click(
                define_world,
                inputs=[world_name, locations, characters],
                outputs=world_status,
            )

            gr.Markdown("### Generate a Story in Your World")
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    story_world = gr.Textbox(label="Enter World Name", placeholder="World name...")
                    event = gr.Textbox(label="Event", placeholder="Describe an event in the world...")
                    selected_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model", type="value")
                    max_length = gr.Slider(50, 300, value=150, step=1, label="Max Length")
    generate_story_button.click(
        generate_story,
        inputs=[selected_model, story_world, max_length, event],
        outputs=generated_story,
    )

    gr.Markdown("Made by **AliDev2020** with ❤️")

# Launch the interface
interface.queue().launch(
    server_port=7860, 
    show_error=True, 
    inline=False,
    #share=True,
)