from fastapi import FastAPI
from .routes import router
from ..core.settings import Settings

settings = Settings()
app = FastAPI(title=settings.PROJECT_NAME)

# Add routes with prefix
app.include_router(router, prefix=settings.API_V1_STR) 