import os, json
from pathlib import Path
from datetime import datetime
from datetime import date
timestamp = datetime.now().strftime("%m%d%H%M")

# Create the json files with model args, data args and training args for the experiments.
# First, set a default
# Then provide the lists of options
# Then provide any manual additional settings

CONFIG_ROOT = "configs"

# Use same keys here as in ds:
label_col = {"tsa": "tsa_tags",
    "ner": "ner_tags"}

# Settings dependent on where we are working
# These must be set manually:
ms, ds, local_out_dir = None, None, None
ltg = "ltg/norbert3-base"
pre_trained_best_tsa = f"/fp/projects01/ec30/liljacs/Master/modeling/output_{date.today()}"
WHERE = "fox"


if WHERE == "fox":
    
    ms = {"NorBERT_3_base": pre_trained_best_tsa}
    local_out_dir = f"/fp/projects01/ec30/liljacs/Master/modeling/output_{date.today()}"
    ds = {"tsa": "/fp/projects01/ec30/liljacs/Master/modeling/data/tsa_arrow_"}


assert not any([e is None for e in [ms, ds, local_out_dir]]), "ms, ds, and local_out_dir need values set above here"
# Add training args as needed
default = {
    "model_name_or_path": None, #ms["brent0"] ,
    "dataset_name": None,
    "per_device_train_batch_size": 32,
    "task_name": "tsa", # Change this in iteration if needed
    "output_dir": local_out_dir,   # Add to this in iteration to avoid overwriting
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "do_train": False,
    "num_train_epochs": 10,
    "do_eval": True,
    "return_entity_level_metrics": False, # True,
    "use_auth_token": False,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch", # 
    "save_total_limit": 1,
    "load_best_model_at_end": True, #  
    "label_column_name": None,
    "disable_tqdm": True,
    "do_predict": True,
    "text_column_name": "tokens"
}


# Iterations: design this according to needs
for task in ds.keys():
    experiments = [] # List of dicts, one dict per experiments: Saves one separate json file each
    for m_name, m_path in ms.items():
        for sd in [101]:#,202,303,404,505]:
            exp = default.copy()
            exp["model_name_or_path"] = m_path
            exp["dataset_name"] = ds [task]
            exp["task_name"] = f"{timestamp}_{task}_{m_name}" # Add seed in name if needed
            exp["output_dir"] = os.path.join(default["output_dir"], exp["task_name"] )
            exp["label_column_name"] = label_col.get(task, "")
            exp["seed"] = sd

            experiments.append(exp)

for i, exp in enumerate(experiments): # Move this with the experiments list definition to make subfolders
    print(exp["task_name"])
    save_path = Path(CONFIG_ROOT, exp["task_name"]+".json")
    save_path.parent.mkdir( parents=True, exist_ok=True)
    with open(save_path, "w") as wf:
        json.dump(exp, wf)