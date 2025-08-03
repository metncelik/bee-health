from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
from fastapi import HTTPException
from routes.predict import router as predict_router

app = FastAPI(
    title="Bee Health API",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

@app.exception_handler(ValueError)
async def value_error_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation Error", "detail": str(exc)}
    )

@app.exception_handler(sqlite3.IntegrityError)
async def sqlite_integrity_error_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation Error", "detail": str(exc)}
    )
    
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )

app.include_router(predict_router)

@app.get("/")
async def root():
    return {
        "title": "Bee Health",
        "version": "0.1.0",
        "docs": "/docs",
    }
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7271)
