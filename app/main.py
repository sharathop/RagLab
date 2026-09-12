import os
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base, check_db_connection
from app.db.models import *  # Ensure all models are registered
from app.routes.web import router as web_router, templates
from app.routes.api import router as api_router
from app.services.session_manager import SessionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist
    try:
        Base.metadata.create_all(bind=engine)
        print("Database schema verified / created.")
    except Exception as e:
        print(f"Database initialization warning: {e}")

    yield

    # Shutdown: clean up in-memory vector stores
    print("Shutting down Self-Correcting RAG application...")


app = FastAPI(
    title="RagLab",
    description="Configurable RAG & Multi-Metric Evaluation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(web_router)
app.include_router(api_router)


# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # If HTML request, render error alert partial
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        html = templates.TemplateResponse(
            "partials/error_alert.html",
            {
                "request": request,
                "error_title": "Internal Error",
                "error_message": f"An unhandled error occurred: {str(exc)}"
            }
        ).body
        return HTMLResponse(content=html, status_code=500)

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error": "InternalServerError"}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
