"""Main file for the backend. Where the endpoints of the borehole ML application are defined."""

import os

import app.common.log as log
from app.api.v1.router import router as v1_router
from app.common.log import get_app_logger
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from mangum import Mangum

# Set up logging
log.setup_logging()
logger = get_app_logger()

load_dotenv()

root_path = os.getenv("ENV", default="stage")
app = FastAPI(root_path=f"/{root_path}")


def custom_openapi():
    """Custom function to modify the OpenAPI schema and remove 422 errors."""
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            terms_of_service=app.terms_of_service,
            contact=app.contact,
            license_info=app.license_info,
            routes=app.routes,
            tags=app.openapi_tags,
            servers=app.servers,
        )
        for _, method_item in app.openapi_schema.get("paths").items():
            for _, param in method_item.items():
                responses = param.get("responses")
                # remove 422 response, also can remove other status code
                if "422" in responses:
                    del responses["422"]
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    try:
        content = {"detail": exc._errors[0]["ctx"]["error"].args[0]}
    except (IndexError, KeyError):
        content = {"detail": "{} field - {}".format(exc.errors()[0]["loc"][1], exc.errors()[0]["msg"])}
    return JSONResponse(content=content, status_code=status.HTTP_400_BAD_REQUEST)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DEL", "PATCH", "PUT"],
)


####################################################################################################
### Health Check
####################################################################################################
@app.get("/health", tags=["health"])
def get_health():
    """Check the health of the application."""
    return {"status": "ok"}


####################################################################################################
### Router
####################################################################################################
logger.debug("Including router in FastAPI app...")
app.include_router(v1_router)

# Allows the integration with AWS Lambda
handler = Mangum(app)
