"""Matching evaluation endpoints."""

from fastapi import APIRouter, Depends, Query

from app.middleware.service_auth import require_service_token
from app.models.common_models import StandardResponse
from app.models.matching_models import EvaluateRequest
from app.services.matching_service import MatchingService

router = APIRouter(prefix="/matching", tags=["Matching"], dependencies=[Depends(require_service_token)])


@router.post("/evaluate")
async def evaluate_match(request: EvaluateRequest):
    """Evaluate a student-program or student-scholarship match."""
    service = MatchingService()
    result = await service.evaluate(
        user_id=request.user_id,
        entity_type=request.entity_type.value,
        entity_id=request.entity_id,
        user_profile=request.user_profile,
        entity_data=request.entity_data,
        include_attribution=request.include_attribution,
    )
    return StandardResponse(success=True, message="Evaluation complete", data=result)


@router.get("/results/{user_id}")
async def get_user_results(
    user_id: str,
    entity_type: str | None = Query(default=None, description="Filter by 'program' or 'scholarship'"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Retrieve paginated match results for a user."""
    service = MatchingService()
    data = await service.get_results_by_user(user_id, entity_type, page, page_size)
    return StandardResponse(success=True, message="OK", data=data)


@router.get("/results/detail/{match_id}")
async def get_match_detail(match_id: str):
    """Retrieve a single match result by ID."""
    service = MatchingService()
    result = await service.get_result_by_id(match_id)
    if not result:
        return StandardResponse(success=False, message="Match result not found", data=None)
    return StandardResponse(success=True, message="OK", data=result)
