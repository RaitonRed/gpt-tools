from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from model import model_dict

app = FastAPI(
    title="GPT Tools API",
    description="API for various GPT-based tools including text generation, code generation, and more.",
    version="1.0.0",
)

class TextGenerationRequest(BaseModel):
    input_text: str
    selected_model: str
    max_new_tokens: int


class CodeGenerationRequest(BaseModel):
    code_prompt: str
    max_tokens: int
    selected_model: str


class MultiverseStoryRequest(BaseModel):
    input_text: str
    selected_model: str
    max_new_tokens: int


class InteractiveStoryRequest(BaseModel):
    input_text: str
    selected_model: str
    max_length: int


class WorldCreationRequest(BaseModel):
    world_name: str
    locations: str
    characters: str


class StoryGenerationRequest(BaseModel):
    model: str
    world_name: str
    event: str
    max_length: int


class ChatbotRequest(BaseModel):
    username: str
    input_text: str
    selected_model: str
    chat_id: Optional[str] = None


class SummarizationRequest(BaseModel):
    input_text: str
    selected_model: str
    max_length: int
    min_length: int


class TranslationRequest(BaseModel):
    input_text: str
    selected_model: str
    mode: str
    max_length: int


class EntertainmentContentRequest(BaseModel):
    topic: str
    content_type: str
    selected_model: str
    max_length: int


@app.post("/generate-text")
async def generate_text_endpoint(request: TextGenerationRequest):
    """
    Generate text based on the input text and selected model.
    """
    from functions import generate
    try:
        output_text = generate(request.input_text, request.selected_model, request.max_new_tokens)
        return {"generated_text": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-code")
async def generate_code_endpoint(request: CodeGenerationRequest):
    """
    Generate code based on the code prompt and selected model.
    """
    from functions import _generate_code
    try:
        generated_code = _generate_code(request.code_prompt, request.max_tokens, request.selected_model)
        return {"generated_code": generated_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-multiverse-story")
async def generate_multiverse_story_endpoint(request: MultiverseStoryRequest):
    """
    Generate parallel universe stories based on the input text.
    """
    from functions import generate_multiverse
    try:
        output_text = generate_multiverse(request.input_text, request.selected_model, request.max_new_tokens)
        return {"generated_worlds": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interactive-story")
async def interactive_story_endpoint(request: InteractiveStoryRequest):
    """
    Generate the next part of an interactive story.
    """
    from functions import interactive_story
    try:
        story_text = interactive_story(request.input_text, request.selected_model, request.max_length)
        return {"story_so_far": story_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-story")
async def reset_story_endpoint():
    """
    Reset the interactive story.
    """
    from functions import reset_story
    try:
        reset_story()
        return {"message": "Story reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-world")
async def create_world_endpoint(request: WorldCreationRequest):
    """
    Create a story world with locations and characters.
    """
    from functions import define_world
    try:
        world_status = define_world(request.world_name, request.locations, request.characters)
        return {"world_status": world_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-story-in-world")
async def generate_story_in_world_endpoint(request: StoryGenerationRequest):
    """
    Generate a story within a predefined world.
    """
    from functions import generate_story
    try:
        generated_story = generate_story(request.model, request.world_name, request.event, request.max_length)
        return {"generated_story": generated_story}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chatbot")
async def chatbot_endpoint(request: ChatbotRequest):
    """
    Interact with a chatbot that can detect emotions.
    """
    from functions import chatbot_response_with_emotion
    try:
        chat_output, chat_id, emotion_output = chatbot_response_with_emotion(request.username, request.input_text,
                                                                             request.selected_model, request.chat_id)
        return {"chat_history": chat_output, "chat_id": chat_id, "detected_emotion": emotion_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-text")
async def summarize_text_endpoint(request: SummarizationRequest):
    """
    Summarize the input text using the selected model.
    """
    from functions import handle_summarization
    try:
        summary_output = handle_summarization(request.input_text, request.selected_model, request.max_length,
                                              request.min_length)
        return {"summary": summary_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate-text")
async def translation_text_endpoint(request: TranslationRequest):
    """
    Translate the input text using the selected model and mode.
    """
    from functions import handle_translation
    try:
        translate_output = handle_translation(request.input_text, request.selected_model, request.mode,
                                              request.max_length)
        return {"translate": translate_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/models",
    summary="Get Available Models",
    description="Retrieve a list of all available models that can be used for text generation, code generation, and other tasks.",
    response_description="A list of model names.",
)
async def get_models():
    """
    Retrieve a list of all available models.
    """
    try:
        # Extract model names from model_dict
        models_list = list(model_dict.keys())
        return {"models": models_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-entertainment-content")
async def generate_entertainment_content_endpoint(request: EntertainmentContentRequest):
    """
    Generate entertainment content (jokes, stories, riddles, poems) based on the topic.
    """
    from functions import generate_entertainment_content
    try:
        output_text = generate_entertainment_content(request.topic, request.content_type, request.selected_model,
                                                     request.max_length)
        return {"generated_content": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)