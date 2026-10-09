from text_summarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from text_summarizer.utils.common import read_yaml, create_directories
from box import ConfigBox
from text_summarizer.entity import (DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainingConfig, ModelEvaluationConfig)

from pathlib import Path
class ConfigurationManager:

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = ConfigBox({
            "root_dir": config.root_dir,
            "source_URL": config.source_URL,
            "local_data_file": config.local_data_file,
            "unzip_dir": config.unzip_dir
        })

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )
        return data_validation_config    
    def get_data_transformation_config(self) -> DataTransformationConfig:
            config = self.config.data_transformation
            create_directories([config.root_dir])
            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                data_path=config.data_path,
                tokenizer_name=config.tokenizer_name
            )
            return data_transformation_config


    def get_model_training_config(self) -> ModelTrainingConfig:

        model_training_config = self.config.model_training
        model_training_params = self.params.model_training

        model_training_config = ModelTrainingConfig(
            root_dir=Path(model_training_config.root_dir),
            data_path=Path(model_training_config.data_path),
            model_ckpt=model_training_config.model_ckpt,
            num_train_epochs=model_training_params.num_train_epochs,
            warmup_steps=model_training_params.warmup_steps,
            per_device_train_batch_size=model_training_params.per_device_train_batch_size,
            weight_decay=model_training_params.weight_decay,
            logging_steps=model_training_params.logging_steps,
            evaluation_strategy=model_training_params.evaluation_strategy,
            eval_steps=model_training_params.eval_steps,
            save_steps=model_training_params.save_steps,
            gradient_accumulation_steps=model_training_params.gradient_accumulation_steps
        )

        return model_training_config
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            data_path=Path(config.data_path),
            tokenizer_path=Path(config.tokenizer_path),
            metric_filename=Path(config.metric_filename)
        )

        return model_evaluation_config    