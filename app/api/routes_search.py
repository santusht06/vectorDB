"""
VectorForge — Search Route

POST /search — Search nearest neighbors using brute, ivf, or hnsw index
"""

from fastapi import APIRouter, HTTPException, Request, status
import numpy as np

from app.core.exceptions import (
    DimensionMismatchError,
    IndexNotBuiltError,
    InvalidSearchParameterError,
)
from app.schemas.response import SearchResponse, SearchResultItem
from app.schemas.search import SearchRequest

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_vectors(req: SearchRequest, request: Request):
    """
    Search for nearest neighbors using the selected index.

    Query parameters inside body:
    - query: list[float]
    - k: int (default 10)
    - index: "brute", "ivf", or "hnsw"
    - nprobe: int (optional, for IVF)
    - ef_search: int (optional, for HNSW)
    """
    manager = request.app.state.manager

    # Build kwargs for search
    kwargs = {}
    if req.nprobe is not None:
        kwargs["nprobe"] = req.nprobe
    if req.ef_search is not None:
        kwargs["ef_search"] = req.ef_search

    try:
        query_vec = np.array(req.query, dtype=np.float32)
        results = manager.search(
            query=query_vec,
            k=req.k,
            index=req.index,
            **kwargs,
        )

        items = [SearchResultItem(id=res.id, score=res.score) for res in results]
        return SearchResponse(
            index=req.index,
            results=items,
            total=len(items),
        )
    except InvalidSearchParameterError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DimensionMismatchError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
