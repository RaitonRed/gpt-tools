from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functions import generate, _generate_code, generate_multiverse, interactive_story, reset_story, define_world, generate_story, chatbot_response_with_emotion, handle_summarization, handle_translation
from typing import Optional

app = FastAPI()

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

@app.post("/generate-text")
async def generate_text_endpoint(request: TextGenerationRequest):
    try:
        output_text = generate(request.input_text, request.selected_model, request.max_new_tokens)
        return {"generated_text": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-code")
async def generate_code_endpoint(request: CodeGenerationRequest):
    try:
        generated_code = _generate_code(request.code_prompt, request.max_tokens, request.selected_model)
        return {"generated_code": generated_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-multiverse-story")
async def generate_multiverse_story_endpoint(request: MultiverseStoryRequest):
    try:
        output_text = generate_multiverse(request.input_text, request.selected_model, request.max_new_tokens)
        return {"generated_worlds": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interactive-story")
async def interactive_story_endpoint(request: InteractiveStoryRequest):
    try:
        story_text = interactive_story(request.input_text, request.selected_model, request.max_length)
        return {"story_so_far": story_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-story")
async def reset_story_endpoint():
    try:
        reset_story()
        return {"message": "Story reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-world")
async def create_world_endpoint(request: WorldCreationRequest):
    try:
        world_status = define_world(request.world_name, request.locations, request.characters)
        return {"world_status": world_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-story-in-world")
async def generate_story_in_world_endpoint(request: StoryGenerationRequest):
    try:
        generated_story = generate_story(request.model, request.world_name, request.event, request.max_length)
        return {"generated_story": generated_story}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbot")
async def chatbot_endpoint(request: ChatbotRequest):
    try:
        chat_output, chat_id, emotion_output = chatbot_response_with_emotion(request.username, request.input_text, request.selected_model, request.chat_id)
        return {"chat_history": chat_output, "chat_id": chat_id, "detected_emotion": emotion_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-text")
async def summarize_text_endpoint(request: SummarizationRequest):
    try:
        summary_output = handle_summarization(request.input_text, request.selected_model, request.max_length, request.min_length)
        return {"summary": summary_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/translate-text")
async def translation_text_endpoint(request: TranslationRequest):
    try:
        translate_output = handle_translation(request.input_text, request.selected_model, request.mode, request.max_length)
        return {"translate": translate_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)