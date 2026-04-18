"""Matching evaluation endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.middleware.service_auth import require_service_token
from app.models.common_models import StandardResponse
from app.models.matching_models import BatchEvaluateRequest, EvaluateRequest
from app.services.matching_service import MatchingService

router = APIRouter(prefix="/matching", tags=["Matching"], dependencies=[Depends(require_service_token)])
internal_router = APIRouter(
    prefix="/api/v1/eligibility",
    tags=["Matching"],
    dependencies=[Depends(require_service_token)],
)


async def _handle_evaluate(request: EvaluateRequest):
    """Evaluate a single student-entity match."""
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


async def _handle_evaluate_batch(request: BatchEvaluateRequest):
    """Evaluate multiple matches for the same student profile."""
    service = MatchingService()
    result = await service.evaluate_batch(
        user_id=request.user_id,
        user_profile=request.user_profile,
        evaluations=[
            {
                "entity_type": item.entity_type.value,
                "entity_id": item.entity_id,
                "entity_data": item.entity_data,
                "include_attribution": item.include_attribution,
            }
            for item in request.evaluations
        ],
        include_attribution=request.include_attribution,
    )
    return StandardResponse(success=True, message="Batch evaluation complete", data=result)


@router.post("/evaluate")
async def evaluate_match(request: EvaluateRequest):
    """Evaluate a student-program or student-scholarship match."""
    return await _handle_evaluate(request)


@router.post("/evaluate/batch")
async def evaluate_match_batch(request: BatchEvaluateRequest):
    """Evaluate multiple matches for the same student profile."""
    return await _handle_evaluate_batch(request)


@internal_router.post("/evaluate", include_in_schema=False)
async def evaluate_internal_match(request: EvaluateRequest):
    """Internal orchestrator-facing alias for single evaluation."""
    return await _handle_evaluate(request)


@internal_router.post("/evaluate/batch", include_in_schema=False)
async def evaluate_internal_match_batch(request: BatchEvaluateRequest):
    """Internal orchestrator-facing alias for batched evaluation."""
    return await _handle_evaluate_batch(request)


@router.get("/results/{user_id}")
async def get_user_results(
    user_id: str,
    entity_type: Optional[str] = Query(default=None, description="Filter by 'program' or 'scholarship'"),
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
