from text_summarizer.logging import logger
from src.text_summarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline


STAGE_NAME = "Data Transformation Stage"

try:

    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")

    data_transformation = DataTransformationTrainingPipeline()

    data_transformation.main()

    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")

except Exception as e:

    logger.exception(e)

    raise e