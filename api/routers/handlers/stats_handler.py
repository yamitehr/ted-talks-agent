from api.models import StatsResponse
import config

async def handle_stats_request() -> StatsResponse:
    return StatsResponse(
        chunk_size=config.CHUNK_SIZE,
        overlap_ratio=float(config.CHUNK_OVERLAP) / float(config.CHUNK_SIZE),
        top_k=config.TOP_K
    )
