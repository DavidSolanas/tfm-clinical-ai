from pathlib import Path

import mlflow
from datasets import DatasetDict
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer

# Imported after unsloth to preserve the patching order (see DECISIONS.md 2026-03-24).
import torch

from src.logging_config import get_logger
from src.tracking import mlflow_run

logger = get_logger(__name__)

_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def finetune(
    dataset: DatasetDict,
    output_dir: str | Path,
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: tuple[str, ...] = _DEFAULT_TARGET_MODULES,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    seed: int = 42,
    mlflow_experiment: str = "tfm-finetuning",
) -> None:
    logger.info(
        "Fine-tuning started — model: %s, epochs: %d, lr: %s, lora_rank: %d, output: %s",
        base_model,
        num_train_epochs,
        learning_rate,
        lora_rank,
        output_dir,
    )

    # bf16 on Ampere+ (e.g. RTX 3090) is numerically stable for long-context
    # training; fp16 needs loss scaling and can NaN. Fall back to fp16 only where
    # bf16 is unsupported. Auto-detect so the same code is correct on any GPU.
    use_bf16 = torch.cuda.is_bf16_supported()

    with mlflow_run(
        experiment=mlflow_experiment,
        run_name="qlora-training",
        tags={"stage": "finetuning", "base_model": base_model},
    ):
        # Log QLoRA and model params that HuggingFace Trainer does not know about.
        # TrainingArguments params (lr, epochs, batch size, etc.) are auto-logged
        # by the HF MLflowCallback when trainer.train() is called.
        mlflow.log_params({
            "base_model": base_model,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": ",".join(target_modules),
            "max_seq_length": max_seq_length,
            "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
            "quantization": "4bit",
            "precision": "bf16" if use_bf16 else "fp16",
        })
        mlflow.log_metrics({
            "train_dataset_size": len(dataset["train"]),
            "val_dataset_size": len(dataset["validation"]),
        })

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=list(target_modules),
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

        def _format(batch: dict) -> dict:
            return {
                "text": [
                    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    for msgs in batch["messages"]
                ]
            }

        train_ds = dataset["train"].map(_format, batched=True)
        val_ds = dataset["validation"].map(_format, batched=True)

        # The HF MLflowCallback detects the active run started above and joins it
        # (_auto_end_run = False), so it logs training metrics here without closing
        # the run — the mlflow_run context manager closes it on exit.
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=SFTConfig(
                output_dir=str(output_dir),
                dataset_text_field="text",
                eos_token="<|eot_id|>",
                max_length=max_seq_length,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                logging_steps=logging_steps,
                fp16=not use_bf16,
                bf16=use_bf16,
                optim="adamw_8bit",
                seed=seed,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                report_to="mlflow",
            ),
        )

        trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Fine-tuning complete — model saved to %s", output_dir)
