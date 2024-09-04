"""Main file for the backend. Where the endpoints of the borehole ML application are defined."""

import app.common.log as log
from app.api.v1.router import router as v1_router
from app.common.log import get_app_logger
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Set up logging
log.setup_logging()
logger = get_app_logger()


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    content = {"detail": exc._errors[0]["ctx"]["error"].args[0]}
    return JSONResponse(content=content, status_code=status.HTTP_400_BAD_REQUEST)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "DEL", "PATCH", "PUT"],
)

logger.debug("Including router in FastAPI app...")
app.include_router(v1_router)

# Optionally, include more routers for other versions or parts of the API
