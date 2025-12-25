from api.models import PromptRequest, PromptResponse, ContextChunk, AugmentedPrompt
from logic.agents.rag_agent import rag_agent
import config

async def handle_prompt_request(request: PromptRequest) -> PromptResponse:
    question = request.question
    
    # Use the agent to get the answer
    result = await rag_agent.aask(question, top_k=config.TOP_K)
    
    # Unpack result
    response_text = result["response_text"]
    docs_and_scores = result["docs_and_scores"]
    system_msg = result["system_msg"]
    user_msg = result["user_msg"]
    
    # Format context for API response
    context_response = []
    for doc, score in docs_and_scores:
        meta = doc.metadata
        context_response.append(ContextChunk(
            talk_id=str(meta.get("talk_id", "N/A")),
            title=meta.get("title", "N/A"),
            chunk=doc.page_content,
            score=score
        ))
        
    return PromptResponse(
        response=response_text,
        context=context_response,
        Augmented_prompt=AugmentedPrompt(
            System=system_msg,
            User=user_msg
        )
    )
