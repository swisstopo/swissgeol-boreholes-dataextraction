"""Main file for the backend. Where the endpoints of the borehole ML application are defined."""

import os

import app.common.log as log
from app.api.v1.router import router as v1_router
from app.common.log import get_app_logger
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

# Set up logging
log.setup_logging()
logger = get_app_logger()

load_dotenv()

root_path = os.getenv("ENV", default="stage")
app = FastAPI(root_path=f"/{root_path}")


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
