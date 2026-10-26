"""The base every response model derives from, and its strictness.

`dw/server/api_models.py` holds the contract this enforces and re-exports
every name here; import from it, not from this module.
"""

import logging
import os

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


STRICT = os.environ.get("DW_STRICT_RESPONSES") == "1"

logger = logging.getLogger("dw.server.api_models")


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid" if STRICT else "allow")


def send_rejected_responses(app) -> None:
    """Make runtime leniency cover a declared field of the wrong type too,
    not just an undeclared key: the response is logged and sent as the
    handler built it, with the route's own status. Some routes have acted
    by then - POST /api/jobs has queued the job - and a 500 would invite a
    retry that does it twice. Installed by create_app outside strict mode."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import ResponseValidationError
    from fastapi.responses import JSONResponse

    async def send_as_built(request, exc: ResponseValidationError):
        route = request.scope.get("route")
        logger.error(
            "Response for %s %s does not match its model: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            jsonable_encoder(exc.body),
            status_code=getattr(route, "status_code", None) or 200,
        )

    app.add_exception_handler(ResponseValidationError, send_as_built)


def sometimes(description: str | None = None) -> Any:
    """A key the handler emits only in some states, and never as null when
    it does: typed as its value, defaulting to an unvalidated None that
    `response_model_exclude_unset` keeps out of the payload. The generated
    type is then `key?: T`, not `key?: T | null`."""
    return Field(default=None, description=description)
