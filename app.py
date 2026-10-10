import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from text_summarizer.pipeline.prediction import PredictionPipeline


# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_summarizer_api")


# =============================
# FastAPI Application
# =============================
app = FastAPI(
    title="Pegasus Text Summarization API",
    description="Generate summaries using a trained Pegasus model",
    version="1.1.0",
)


# =============================
# Load Model Once
# =============================
logger.info("Loading the prediction pipeline...")
prediction_pipeline = PredictionPipeline()
logger.info("Prediction pipeline loaded successfully.")


# =============================
# Home
# =============================
@app.get("/", tags=["Home"])
def index():
    return RedirectResponse(url="/docs")


# =============================
# Health Check (used by the UI "API online" badge)
# =============================
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": prediction_pipeline is not None,
    }


# =============================
# Predict
# =============================
@app.post("/predict", tags=["Prediction"])
def predict(
    text: str = Query(..., min_length=1, max_length=20000, description="Text to summarize"),
):
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    try:
        summary = prediction_pipeline.predict(clean_text)  # change if your method name differs
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Summarization failed. Check the server logs.",
        )

    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)