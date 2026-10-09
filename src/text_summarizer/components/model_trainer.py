from datasets import load_dataset, load_from_disk
from transformers import  DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,Trainer, TrainingArguments, AutoModelForSeq2SeqLM,AutoTokenizer
import os
import torch
from text_summarizer.entity import ModelTrainingConfig
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)



class ModelTrainer:

    def __init__(self, config: ModelTrainingConfig):

        self.config = config

        # ============================================================
        # 1. Load tokenizer
        # ============================================================
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_ckpt
        )

        # ============================================================
        # 2. Load Pegasus base model
        # ============================================================
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        )

        # ============================================================
        # 3. Important for Gradient Checkpointing
        # ============================================================
        self.model.config.use_cache = False

        # ============================================================
        # 4. LoRA Configuration
        # ============================================================
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
        )

        # ============================================================
        # 5. Add LoRA adapters to Pegasus
        # ============================================================
        self.model = get_peft_model(
            self.model,
            lora_config
        )

        # ============================================================
        # 6. Print trainable parameters
        # ============================================================
        self.model.print_trainable_parameters()

        # ============================================================
        # 7. Data Collator
        # ============================================================
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

    # ================================================================
    # TRAIN
    # ================================================================
    def train(self):

        # ============================================================
        # 1. Check GPU
        # ============================================================
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"VRAM: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )

        # ============================================================
        # 2. Load tokenized dataset
        # ============================================================
        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        print(dataset_samsum_pt)

        # ============================================================
        # 3. Training Arguments
        # ============================================================
        training_args = Seq2SeqTrainingArguments(

            # Output
            output_dir=self.config.root_dir,

            # Training
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,

            # Optimization
            learning_rate=5e-4,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=16,

            # Logging
            logging_steps=self.config.logging_steps,
            logging_strategy="steps",

            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,

            # Saving
            save_strategy="steps",
            save_steps=1000000,
            save_total_limit=1,

            # Memory optimization
            fp16=True,
            gradient_checkpointing=True,

            # Optimizer
            optim="adamw_torch",

            # Disable unnecessary reporting
            report_to="none",

            # Prevent unused column problems
            remove_unused_columns=True,
        )

        # ============================================================
        # 4. Create Trainer
        # ============================================================
        trainer = Seq2SeqTrainer(

            model=self.model,

            args=training_args,

            train_dataset=dataset_samsum_pt["train"],

            eval_dataset=dataset_samsum_pt["validation"],

            processing_class=self.tokenizer,

            data_collator=self.data_collator,
        )

        # ============================================================
        # 5. Start Training
        # ============================================================
        print("\n========== Starting Training ==========\n")

        trainer.train()

        # ============================================================
        # 6. Save LoRA Adapter
        # ============================================================
        model_output_path = os.path.join(
            self.config.root_dir,
            "pegasus_lora"
        )

        self.model.save_pretrained(
            model_output_path
        )

        # ============================================================
        # 7. Save Tokenizer
        # ============================================================
        tokenizer_output_path = os.path.join(
            self.config.root_dir,
            "pegasus_tokenizer"
        )

        self.tokenizer.save_pretrained(
            tokenizer_output_path
        )

        print("\n========== Training Completed ==========")

        print(
            f"LoRA model saved at: {model_output_path}"
        )

        print(
            f"Tokenizer saved at: {tokenizer_output_path}"
        )
