import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_summarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Prediction device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(self.device)

        self.model.eval()
        print("Prediction model loaded successfully.")

    def predict(self, dialogue: str) -> str:
        if not dialogue or not dialogue.strip():
            raise ValueError("Dialogue cannot be empty.")

        inputs = self.tokenizer(
            "summarize: " + dialogue.strip(),
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            summary_ids = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=6,
                length_penalty=0.8,
                early_stopping=True,
            )

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
