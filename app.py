
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
    version="1.0.0",
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
# Health Check
# =============================
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": prediction_pipeline is not None,
    }


# =============================
# Prediction Endpoint
# =============================
@app.post("/predict", tags=["Prediction"])
def predict(
    text: str = Query(
        ...,
        min_length=1,
        description="Text to summarize",
    )
):
    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Please enter text to summarize.",
        )

    try:
        logger.info(
            "Received text with %d characters.",
            len(text),
        )

        summary = prediction_pipeline.predict(text)

        logger.info("Summary generated successfully.")

        return {
            "summary": summary,
        }

    except Exception as e:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        ) from e


# =============================
# Run Server
# =============================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
