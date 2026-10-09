from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset          # شلنا load_from_disk و load_metric
import evaluate                             # 👈 المكتبة الجديدة
import torch
import pandas as pd
from tqdm import tqdm
from text_summarizer.entity import ModelEvaluationConfig
import importlib

hf_evaluate = importlib.import_module("evaluate")
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]


    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer, 
                               batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                               column_text="article", 
                               column_summary="highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
                            padding="max_length", return_tensors="pt")
            
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device), 
                            length_penalty=0.8, num_beams=8, max_length=128)
            
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True) 
                for s in summaries]      
            
            decoded_summaries = [d.replace("", "") for d in decoded_summaries]
            
            
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        score = metric.compute()
        return score



    
    def evaluate(self):
        # 1. Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 2. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        # 3. Load trained Pegasus model
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)

        model_pegasus.eval()

        # 4. Load SAMSum CSV files
        dataset_samsum = load_dataset(
            "csv",
            data_files={
                "train": "artifacts/data_ingestion/samsum-train.csv",
                "validation": "artifacts/data_ingestion/samsum-validation.csv",
                "test": "artifacts/data_ingestion/samsum-test.csv",
            },
        )

        # 5. Load ROUGE metric
        rouge_names = [
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ]

        rouge_metric = hf_evaluate.load("rouge")

        # 6. Evaluate on the first 10 test examples
        test_samples = dataset_samsum["test"].select(
            range(min(10, len(dataset_samsum["test"])))
        )

        score = self.calculate_metric_on_test_ds(
            test_samples,
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )

        # 7. Convert ROUGE scores into a dictionary
        rouge_dict = {}

        for rn in rouge_names:
            value = score[rn]

            if hasattr(value, "mid"):
                rouge_dict[rn] = float(value.mid.fmeasure)
            else:
                rouge_dict[rn] = float(value)

        # 8. Print results
        print("\nROUGE Evaluation Results:")

        for metric_name, metric_value in rouge_dict.items():
            print(f"{metric_name}: {metric_value:.4f}")

        # 9. Save results to CSV
        df = pd.DataFrame(
            [rouge_dict],
            index=["pegasus"],
        )

        df.to_csv(
            self.config.metric_filename,
            index=False,
        )

        print(
            f"\nMetrics saved to: "
            f"{self.config.metric_filename}"
        )

        return rouge_dict
