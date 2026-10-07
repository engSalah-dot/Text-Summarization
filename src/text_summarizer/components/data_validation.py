import os
import src.text_summarizer.logging

from text_summarizer.entity import DataValidationConfig
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_required_files(self) -> bool:
        try:
            validation_status = None
            all_files=os.listdir(os.path.join("artifacts","data_ingestion","samsum_dataset"))
            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Required files are not present. Missing file: {file}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write("All required files are present.")
            return validation_status
                    
        except Exception as e:
            src.text_summarizer.logging.error(f"Error occurred while validating required files: {e}")
            raise e
                                

                
              

       