"""Attribution / explainability report endpoints."""

from fastapi import APIRouter, Depends

from app.core.logging import get_logger
from app.middleware.service_auth import require_service_token
from app.models.common_models import StandardResponse
from app.services.explainability_service import ExplainabilityService

router = APIRouter(prefix="/attribution", tags=["Attribution"], dependencies=[Depends(require_service_token)])
internal_router = APIRouter(
    prefix="/api/v1/eligibility",
    tags=["Attribution"],
    dependencies=[Depends(require_service_token)],
)

logger = get_logger(__name__)


@router.get("/report/{match_id}")
async def get_attribution_report(match_id: str):
    """Retrieve the attribution report for a given match evaluation."""
    service = ExplainabilityService()
    try:
        report = await service.get_report(match_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("attribution_report_fetch_failed", match_id=match_id, error=str(exc))
        return StandardResponse(
            success=False,
            message="Unable to fetch attribution report right now",
            data=None,
        )
    if not report:
        return StandardResponse(success=False, message="Attribution report not found", data=None)
    return StandardResponse(success=True, message="OK", data=report)


@internal_router.get("/report/{match_id}", include_in_schema=False)
async def get_internal_eligibility_report(match_id: str):
    """Internal alias used by orchestrator-facing integrations."""
    return await get_attribution_report(match_id)
