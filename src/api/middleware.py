from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

VALID_API_KEY = os.environ.get("API_KEY")
if not VALID_API_KEY:
    raise ValueError("API_KEY environment variable is not set")

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != VALID_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid or missing API key")
        return await call_next(request)
