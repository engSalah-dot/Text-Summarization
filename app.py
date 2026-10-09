
import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from text_summarizer.pipeline.prediction import PredictionPipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_summarizer_api")

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict(
    text: str = Query(..., min_length=1, description="Text to summarize")
):
    try:
        logger.info("Starting prediction pipeline...")

        prediction_pipeline = PredictionPipeline()

        logger.info("Prediction pipeline initialized. Generating summary...")

        summary = prediction_pipeline.predict(text)

        logger.info("Prediction completed successfully.")

        return {"summary": summary}

    except Exception as e:
        # Print the complete traceback in the terminal
        logger.exception("Error occurred during prediction")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
