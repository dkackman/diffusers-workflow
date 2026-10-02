"""Response models for the routes the web UI reads - the server half of the
UI's response contract. The UI's types are generated from the OpenAPI
document these produce (scripts/dump_openapi.py -> ui/src/lib/generated/),
so a change here that the UI depends on fails its type check.

Lenient at runtime: a key a model has not declared is still sent, so a
handler that grows a field never turns into a 500 for dw_mcp or a script.
Strict under DW_STRICT_RESPONSES=1 - the test suite, the e2e fixture server
and the OpenAPI dump - so an undeclared key fails a test, and the generated
types carry no index signature that would let a removed field type-check.

Routes declare `response_model_exclude_unset=True`: a key the handler did
not emit stays absent rather than arriving as null.
"""

import os

from pydantic import BaseModel, ConfigDict

STRICT = os.environ.get("DW_STRICT_RESPONSES") == "1"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid" if STRICT else "allow")
