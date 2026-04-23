"""Attribution / explainability report endpoints."""

from fastapi import APIRouter, Depends

from app.middleware.service_auth import require_service_token
from app.models.common_models import StandardResponse
from app.services.explainability_service import ExplainabilityService

router = APIRouter(prefix="/attribution", tags=["Attribution"], dependencies=[Depends(require_service_token)])
internal_router = APIRouter(
    prefix="/api/v1/eligibility",
    tags=["Attribution"],
    dependencies=[Depends(require_service_token)],
)


@router.get("/report/{match_id}")
async def get_attribution_report(match_id: str):
    """Retrieve the attribution report for a given match evaluation."""
    service = ExplainabilityService()
    report = await service.get_report(match_id)
    if not report:
        return StandardResponse(success=False, message="Attribution report not found", data=None)
    return StandardResponse(success=True, message="OK", data=report)


@internal_router.get("/report/{match_id}", include_in_schema=False)
async def get_internal_eligibility_report(match_id: str):
    """Internal alias used by orchestrator-facing integrations."""
    return await get_attribution_report(match_id)
