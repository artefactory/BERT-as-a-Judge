import json
import os

from datasets import (
    Dataset,
    concatenate_datasets,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..utils import (
    load_hf_encoder,
    load_hf_tokenizer,
)


class BERTJudge:
    def __init__(
        self,
        model_path,
        trust_remote_code=False,
        dtype="bfloat16",
        device_map="auto",
    ):
        self.special_tokens = {
            "question": "<|question|>",
            "candidate": "<|candidate|>",
            "reference": "<|reference|>",
        }
        self.model = load_hf_encoder(
            model_path, 
            trust_remote_code,
            dtype,
            device_map,
        )
        self.max_length = getattr(self.model.config, "max_position_embeddings")
        self.tokenizer = load_hf_tokenizer(
            model_path, 
            trust_remote_code=trust_remote_code,
        )
        self._add_special_tokens()

    def fit(
        self,
        dataset,
        output_dir,
        include_question=True,
        training_mix=None,
        num_train_epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=None,
        report_to=None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        seed=0,
    ):                
        if report_to is None:
            report_to = ["tensorboard"]

        if training_mix:
            dataset = self._apply_training_mix(dataset, training_mix, seed)
        else:            
            dataset = self._flatten_dataset(dataset)

        dataset = self._make_prompts(dataset, include_question)
        dataset = self._tokenize_prompts(dataset)
        trainer = self._build_trainer(
            dataset,
            output_dir,
            num_train_epochs,
            batch_size,
            learning_rate,
            warmup_ratio,
            lr_scheduler_type,
            logging_strategy,
            logging_steps,
            logging_dir,
            report_to,
            save_strategy,
            save_steps,
            save_total_limit,
            seed,
        )
        trainer.train()
        self._save_model(output_dir)

    def predict(
        self,
        questions,
        candidates,
        references,
        batch_size=1,
    ):
        if not questions:
            questions = [""] * len(references)
            include_question = False
        else:
            include_question = True

        dataset = Dataset.from_dict({
            "question": questions,
            "candidate": candidates,
            "reference": references,
        })
        dataset = self._make_prompts(dataset, include_question)
        dataset = self._tokenize_prompts(dataset)
        dataloader = self._build_dataloader(dataset, batch_size)
        self.model.eval()
        
        scores = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                output = self.model(**batch)
                scores += output.logits.cpu().tolist()
        
        return [torch.tensor(s[1] - s[0]).sigmoid().item() for s in scores]
    
    def _add_special_tokens(self):        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.tokenizer.add_tokens(list(self.special_tokens.values()))
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _apply_training_mix(
        self,
        dataset,
        training_mix,
        seed=0,
    ):
        processed_dataset = []
        for name in training_mix:
            for split, num_samples in training_mix[name].items():
                subset = dataset[name][split].shuffle(seed)
                processed_dataset.append(
                    subset.select(range(min(len(subset), num_samples)))
                )

        return concatenate_datasets(processed_dataset)
    
    def _flatten_dataset(
        self, 
        dataset,
    ):
        return concatenate_datasets(
            [dataset[name][split] for name in dataset for split in dataset[name]]
        )

    def _make_prompts(
        self,
        dataset,
        include_question=True,
    ):
        def fn(ex):
            prompt = ""
            if include_question:
                prompt += self.special_tokens["question"] + ex["question"]
            prompt += self.special_tokens["candidate"] + ex["candidate"]
            prompt += self.special_tokens["reference"] + ex["reference"]
            return {"prompt": prompt}
        
        dataset = dataset.map(
            fn,
            num_proc=max(1, (os.cpu_count() or 1) // 2),
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        
        if "label" in dataset.column_names:
            return dataset.select_columns(["prompt", "label"])
        else:
            return dataset.select_columns(["prompt"])
    
    def _tokenize_prompts(
        self,
        dataset,
    ):
        def fn(ex):
            return self.tokenizer(
                ex["prompt"],
                truncation=True,
                max_length=self.max_length,
                truncation_side="left",
            )

        return dataset.map(
            fn,
            num_proc=max(1, (os.cpu_count() or 1) // 2),
            keep_in_memory=True,
            load_from_cache_file=False,
        ).remove_columns(["prompt"])

    def _build_trainer(
        self,
        dataset,
        output_dir,
        num_train_epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=None,
        report_to=None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        seed=0,
    ):
        if report_to is None:
            report_to = ["tensorboard"]

        training_args = TrainingArguments(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            logging_dir=logging_dir,
            report_to=report_to,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            output_dir=output_dir,
            seed=seed,
        )
        data_collator = DataCollatorWithPadding(self.tokenizer)
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

    def _save_model(
        self, 
        output_dir,
    ):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _build_dataloader(
        self, 
        dataset, 
        batch_size,
    ):
        data_collator = DataCollatorWithPadding(self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
