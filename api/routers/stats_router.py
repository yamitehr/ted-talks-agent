from fastapi import APIRouter
from api.models import StatsResponse
from api.routers.handlers.stats_handler import handle_stats_request

router = APIRouter()

@router.get("/stats", response_model=StatsResponse)
async def stats():
    return await handle_stats_request()
