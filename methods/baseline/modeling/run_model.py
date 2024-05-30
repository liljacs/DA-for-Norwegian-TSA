"""
A simplified version of the run_ner.py script from the transformers library, https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification
The script does sequence labeling, aka token classification, on datasets with BIO tags.
Preferred usage is to pass a json file with configuration parameters as the only argument.
Alternatively,  pass individual settings as keyword arguments. In default_params, the arguments that can be passed are defined.
Any ModelArguments, DataTrainingArguments, TrainingArguments may be defined here or in the json file.

For each task, specify path to the dataset in "dataset_name", specify "task_name", "output_dir" and "label_column_name" as well.
label_column_name is "tsa_tags" in the TSA dataset and  "ner_tags" in the NER dataset.

Further modified for Data augmentation for Norwegian Sentiment Analysis thesis.
"""

from datasets import ClassLabel,  load_dataset, load_from_disk, DatasetDict, Dataset
import os, sys, json
import argparse
from pathlib import Path
import evaluate
import transformers
from datetime import date
import numpy as np
import torch
import pandas as pd
import logging
import wandb
from analyse import Analysis
import math

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

# local files import
from tsa_utils import tsa_eval
from tsa_utils import ModelArguments, DataTrainingArguments
from modeling_norbert import NorbertForTokenClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# for wandb visualization
os.environ["WANDB_API_KEY"] = " " # fill in
os.environ["WANDB_PROJECT"] = " " # fill in
# os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "true"

print("Numpy:", np.version.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

# model paths
norbert3_base = "/fp/projects01/ec30/models/norlm/norbert3-base"
norbert3_large = "/fp/projects01/ec30/models/norlm/norbert3-large"
nbbert_large_fox = "/fp/projects01/ec30/models/nb-bert-large"
nbbert_large_saga = "/cluster/projects/nn9851k/models/nb-bert-large"

#model_name = "baseline_polarspan"
#model_name = "rulebased" # add percentage info
model_name = "baseline_dev" # 

logging.info("Finished importing")
hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
# default_params define what named parameters can be sent to the script

default_params = { # Add entries here that you want to change or just track the value of
    "model_name_or_path": nbbert_large_saga,
    "dataset_name": "prompt_augmented/prompt_augmented_arrow", #"augmented_data/tsa_augmented_arrow", "tsa_conll"
    "seed": 101,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 32,
    "task_name": "tsa", 
    "output_dir": f"/cluster/home/liljacs/Master/modeling/output/{model_name}_{date.today()}", 
    "report_to": "wandb", # optional
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "do_train": True,
    "num_train_epochs": 10,
    "do_eval": True,
    "return_entity_level_metrics": False, # Since we use a separate evaluation script, this is better kept False
    "use_auth_token": False,
    "logging_strategy": "epoch", # "epoch", changed to default
    "save_strategy": "no", # "epoch"
    "evaluation_strategy": "epoch", # "epoch". changed to work with wandb
    "save_total_limit": 0,
    "load_best_model_at_end": False, # Evaluate the last epoch 
    "label_column_name": "tsa_tags",
    "disable_tqdm": False,
    "do_predict": True,
    "text_column_name": "tokens",
    "resume_from_checkpoint": False
}

parser=argparse.ArgumentParser(description = "Pass the path to a json file with configuration parameters as positional argument, or pass individual settings as keyword arguments.")
parser.add_argument("config", nargs="?")
for key, value in default_params.items():
    parser.add_argument(f"--{key}", default=value, type=type(value))
args = parser.parse_args()
if args.config is not None:
    with open (os.path.abspath(args.config)) as rf:
        args_dict = json.load(rf)
else:
    args_dict = vars(args)

args_dict.pop("config", None)
# Since we are flexible with what arguments are defined, we need to convert 

label_col = args_dict["label_column_name"]
logging.info(f"training with seed {args_dict['seed']}")

# Result json-file path
json_results_path = f"/cluster/home/liljacs/Master/modeling/output_prompt/json_result_files" # change from baseline to rulebased when needed, with percentage
results_dict = {"model_name":f"{model_name}_{args.seed}"}

# Checking if output path/json results exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(json_results_path, exist_ok=True)

def compute_metrics(pred):
    print("Inside compute_metrics")
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
   
    results = metric.compute(predictions=true_predictions, references=true_labels)
    batista_f1, batista_dict = tsa_eval(true_labels, true_predictions)
    
    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        results_dict["precision"] = results["overall_precision"] # adding to dictionary later to be converted to json
        results_dict["recall"] = results["overall_recall"]
        results_dict["batista_f1"] = batista_f1
        results_dict["accuracy"] = results["overall_accuracy"]

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "seqeval_f1": results["overall_f1"],
            "batista_f1": batista_f1,
            "accuracy": results["overall_accuracy"],
        }

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    sorted_labels = sorted(label_list,key=lambda name: (name[1:], name[0])) # Gather B and I
    return sorted_labels

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None or word_idx == previous_word_idx :
                label_ids.append(-100)
            # We set the label for the first token of each word only.
            else : #New word
                label_ids.append(label_to_id[label[word_idx]])
            # We do not keep the option to label the subsequent subword tokens here.

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

model_args, data_args, training_args = hf_parser.parse_dict(args_dict)

text_column_name = data_args.text_column_name
label_column_name = data_args.label_column_name
assert data_args.label_all_tokens == False, "Our script only labels first subword token"
dsd = load_from_disk(data_args.dataset_name)

transformers.logging.set_verbosity_warning()

label_list = get_label_list(dsd["train"][data_args.label_column_name]) # "tsa_tags", was train_1
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)
labels_are_int = False

logging.info("Setting up config and tokenizer")
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    trust_remote_code=True,
)
tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=True,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    trust_remote_code=True,
)

# %%
# Instanciate the model
logging.info(f"Setting up pretrained model:{model_args.model_name_or_path}")

if "norbert3" in model_args.model_name_or_path:
    model = NorbertForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        trust_remote_code=True,
    )

else:
        model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

print("Our label2id:    ", label_to_id)

assert (model.config.label2id == PretrainedConfig(num_labels=num_labels).label2id) or (model.config.label2id == label_to_id), "Model seems to have been fine-tuned on other labels already. Our script does not adapt to that."

# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}

# Preprocessing the dataset
# Padding strategy
padding = "max_length" if data_args.pad_to_max_length else False

with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = dsd["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file= False,
        desc="Running tokenizer on train dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_dataset = dsd["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    predict_dataset = dsd["validation"].map( # change when testing
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on test dataset",
    )

print("Dataset features are now:", list(train_dataset.features))

data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

# Metrics
metric = evaluate.load("seqeval") # 
# metric = evaluate.evaluator(task = 'token-classification' )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logging.info("Training model")
train_result = trainer.train() # removed resume from checkpoint=False
metrics = train_result.metrics
metrics["train_samples"] =  len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics(f"train_{args.seed}", metrics)

try:
    print("trying to save loss")
    loss_df = pd.DataFrame(trainer.state.log_history)
    loss_list = loss_df["loss"].tolist()
    results_dict["loss"] = [x for x in loss_list if not math.isnan(x)]
    #df = pd.DataFrame(loss)
    #csv_df = df.to_csv(f"{args_dict['output_dir']}/train_loss_{args.seed}.csv")
except Exception as error:
    print("Could not save loss:", type(error), "-", error)


# Evaluate
"""logging.info("Evaluating on dev")
print("\nEvaluation,",model_args.model_name_or_path)
trainer_eval = trainer.evaluate(eval_dataset)
trainer.save_metrics(f"eval_{args.seed}", trainer_eval)"""


logging.info("Predicting on test")
trainer_predict = trainer.predict(predict_dataset, metric_key_prefix="predict")
predictions, labels, metrics = trainer_predict
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

gold = predict_dataset[args_dict["label_column_name"]] # Note: Will not work if dataset has ints, and the text labels in metadata
for g, pred in zip(gold,true_predictions):
    assert len(g) == len(pred), (len(g) , len(pred))

batista_f1, batista_dict = tsa_eval(gold, true_predictions)
try:
     if data_args.return_entity_level_metrics:
         seqeval_f1 =  metrics["predict_overall_f1"]
     else:
         seqeval_f1 =  metrics["predict_f1"]
except:
     seqeval_f1 = 0
     
print("batista_f1",batista_f1)
print("seqeval_f1",seqeval_f1)
# print("Seqeval > Batista:",round(seqeval_f1 - batista_f1, 4))

# Save predictions
output_predictions_file = os.path.join(training_args.output_dir, f"predictions_{model_name}_1_seed_{args.seed}.txt")
with open(output_predictions_file, "w") as writer:
    for prediction in true_predictions:
        writer.write(" ".join(prediction) + "\n")

args_dict["test_f1"] = batista_f1
wandb.log({"test-f1":batista_f1})
wandb.log({"test-accuracy": metrics["predict_accuracy"]})
# saving test metrics to results dict
results_dict["test_batista_f1"] = batista_f1
results_dict["test_recall"] = metrics["predict_recall"]
results_dict["test_precision"] = metrics["predict_precision"]
results_dict["test_accuracy"] = metrics["predict_accuracy"]

analysis = Analysis()
analysis.readfiles(output_predictions_file, "true_dev.txt") # endre ved test 
polar_fscore = analysis.polar_f1()
strict_span, kind_span = analysis.span_fscore()
adjacent_chunks, broken_chunks, mixed_chunks = analysis.summary()

results_dict["test_polar_f1"] = polar_fscore
results_dict["test_strict_span_f1"] = strict_span
results_dict["test_kind_span_f1"] = kind_span
results_dict["test_adjacent_chunks"] = adjacent_chunks
results_dict["test_broken_chunks"] = broken_chunks
results_dict["test_mixed_chunks"] = mixed_chunks

with open(f"{json_results_path}/all_metrics_{model_name}_seed_{args.seed}.json", "w", encoding="utf-8") as json_file:
    json.dump(results_dict, json_file)

try:
    trainer.save_metrics(f"predict_{args.seed}", metrics)
except:
    print("Could not save test metrics")

with open(f"{args_dict['output_dir']}/batista_results_test_seed_{args.seed}.json", "w") as batista_file:
        json.dump(batista_dict, batista_file)

# save_path = os.path.abspath(args_dict["output_dir"])
save_path = Path(args_dict["output_dir"]).resolve()
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(save_path, args_dict["task_name"]+"_results.json").write_text(json.dumps(args_dict))

# remove cache from wandb after every run (NB: Disse sletter visst mye mer, ikke bruk)
# os.system("cd /cluster/home/liljacs/.cache/wandb/artifacts/obj & rm -r *")
