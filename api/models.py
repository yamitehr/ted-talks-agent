from pydantic import BaseModel
from typing import List, Optional

class PromptRequest(BaseModel):
    question: str

class ContextChunk(BaseModel):
    talk_id: str
    title: str
    chunk: str
    score: Optional[float] = None

class AugmentedPrompt(BaseModel):
    System: str
    User: str

class PromptResponse(BaseModel):
    response: str
    context: List[ContextChunk]
    Augmented_prompt: AugmentedPrompt

class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int
