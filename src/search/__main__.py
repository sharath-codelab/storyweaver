"""Run the recommendation API with `python -m src.search`."""

import os

import uvicorn

from .app import create_app


if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("SEARCH_PORT", "8000"))),
    )
