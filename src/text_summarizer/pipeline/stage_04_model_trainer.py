from text_summarizer.components.model_trainer import ModelTrainer
from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.entity import ModelTrainingConfig
from text_summarizer.logging import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()

        model_trainer = ModelTrainer(model_training_config)
        model_trainer.train()
        
        