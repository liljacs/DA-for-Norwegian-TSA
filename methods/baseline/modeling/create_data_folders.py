# Convert the TSA conll data to DatasetDict
import pandas as pd
import os
from datasets import Dataset, DatasetDict
root_folder = "prompt_augmented"
arrow_folder = os.path.join(root_folder,"prompt_augmented_arrow")

def parse_conll(raw:str, sep="\t"):
    """Parses the norec-fine conll files with tab separator and sentence id"""
    doc_parsed = [] # One dict per sentence. meta, tokens and tags
    for sent in raw.strip().split("\n\n"):
        meta = ""
        tokens, tags = [], []
        for line in sent.split("\n"):
            if line.startswith("#") and "=" in line:
                meta = line.split("=")[-1]
            else:
                elems = line.strip().split(sep)
                if len(elems) == 2:
                    tokens.append(elems[0])
                    tags.append(elems[1])
                else:
                    tokens.append(elems[1]) # added else because some files have indices at start
                    tags.append(elems[2])
        assert len(meta) > 0
        doc_parsed.append({"idx": meta, "tokens":tokens, "tsa_tags":tags})

    return doc_parsed

# label_mapping = {'O':0, 'B-targ-Positive':1, 'I-targ-Positive':2, 'B-targ-Negative':3, 'I-targ-Negative':4, }
# change train in splits to create different sizes with augmented data/different prompts
splits = {"train": "train", "dev": "validation", "test": "test"} 

d_sets = {}
for split in splits:
    path = os.path.join(root_folder, split+".conll")
    with open(path) as rf:
        conll_txt = rf.read()
    print(len(conll_txt.split("\n\n")))
    sents = parse_conll(conll_txt)
    # for sent in sents:
        # sent["labels"] = [label_mapping[tag] for tag in sent["tsa_tags"]]
    d_sets[splits[split]] = Dataset.from_pandas(pd.DataFrame(sents))

DatasetDict(d_sets).save_to_disk(arrow_folder)
    # sentences = parse(conll_txt)
    # sentences[0]