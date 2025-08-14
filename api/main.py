from fastapi import FastAPI
from .routes import chat

app = FastAPI()

app.include_router(chat.router, prefix="/v1")
