"""
Entry point — runs the FastAPI server.
"""

import uvicorn
from app.api.routes import app

if __name__ == "__main__":
    uvicorn.run(
        "app.api.routes:app",
        host    = "0.0.0.0",
        port    = 8080,
        reload  = True,
    )