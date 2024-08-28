"""Main file for the backend. Where the endpoints of the borehole ML application are defined."""

import app.common.log as log
from app.api.v1.router import router as v1_router
from app.common.log import get_app_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
log.setup_logging()
logger = get_app_logger()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "DEL", "PATCH", "PUT"],
)

logger.debug("Including router in FastAPI app...")
app.include_router(v1_router)

# Optionally, include more routers for other versions or parts of the API
