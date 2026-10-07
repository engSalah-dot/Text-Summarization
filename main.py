from text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from text_summarizer.logging import logger
from src.text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

STAGE_NAME = "Data Validation Stage"

try:

    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")

    data_validation = DataValidationTrainingPipeline()

    data_validation.main()

    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")

except Exception as e:

    logger.exception(e)

    raise e