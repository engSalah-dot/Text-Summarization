
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from text_summarizer.config.configuration import ConfigurationManager


class PredictionPipeline:

    def __init__(self):
        self.config = (
            ConfigurationManager().get_model_evaluation_config()
        )

        # Select device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        # Load trained model once
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(self.device)

        self.model.eval()

    def predict(self, text: str) -> str:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text cannot be empty.")

        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        # Move input tensors to CPU or GPU
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Preserve the original generation settings
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128,
        }

        # Generate summary
        with torch.inference_mode():
            summary_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode generated tokens
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
        )

        return summary
