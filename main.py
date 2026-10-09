from src.text_summarizer.logging import logger
from src.text_summarizer.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline



STAGE_NAME = "MOdEL EvALUATION"

try:

    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")

    model_evaluation = ModelEvaluationPipeline()

    model_evaluation.main()

    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")

except Exception as e:

    logger.exception(e)

    raise e