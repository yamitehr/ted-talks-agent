from fastapi import FastAPI
from api.routers import prompt_router, stats_router

app = FastAPI()

app.include_router(prompt_router.router, prefix="/api")
app.include_router(stats_router.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "TED Talk RAG Agent API is running."}
