from fastapi import APIRouter
from api.models import PromptRequest, PromptResponse
from api.routers.handlers.prompt_handler import handle_prompt_request

router = APIRouter()

@router.post("/prompt", response_model=PromptResponse)
async def prompt(request: PromptRequest):
    return await handle_prompt_request(request)
