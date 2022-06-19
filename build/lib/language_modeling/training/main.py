from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from language_modeling.training.dataset import get_dataset
from language_modeling.util import load_hparams
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
from pathlib import Path
from transformers import Trainer
from transformers import default_data_collator
from transformers import TrainingArguments
import math
from language_modeling.util import get_default_logger
from language_modeling.Project import Project
from box import Box

logger = get_default_logger()

hparams = load_hparams()

logger.info(f"Using Model {hparams.base_model_name}")
model = AutoModelForMaskedLM.from_pretrained(hparams.base_model_name)
tokenizer = AutoTokenizer.from_pretrained(hparams.base_model_name)

text = "This is a great [MASK]."
dataset = get_dataset()


def prepare_dataset(dataset):
    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    return tokenized_datasets


tokenized_datasets = prepare_dataset(dataset)


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // hparams.chunk_size) * hparams.chunk_size
    # Split by chunks of max_len
    result = {
        k: [
            t[i : i + hparams.chunk_size]
            for i in range(0, total_length, hparams.chunk_size)
        ]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, hparams.wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)


train_size = hparams.train_test_split_ratio
test_size = 1 - hparams.train_test_split_ratio
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size
)

# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // hparams.batch_size
model_name = hparams.base_model_name.split("/")[-1]
output_dir = Project.export_dir / f"{model_name}-finetuned"

logger.info(f"Saving model and tokenizer to {output_dir}")

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=hparams.learning_rate,
    weight_decay=hparams.weight_decay,
    per_device_train_batch_size=hparams.batch_size,
    per_device_eval_batch_size=hparams.batch_size,
    fp16=hparams.fp16,
    logging_steps=logging_steps,
    num_train_epochs=hparams.epochs,
    report_to="wandb",
)


logger.info("Initializing trainer object")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)
tokenizer.save_pretrained(output_dir)

logger.info("Running evaluation before training")
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


logger.info("Running evaluation after training")
trainer.train()
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
