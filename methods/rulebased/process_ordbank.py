import stanza
import json
from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

# Writing the "ordbank" resources to new file with support for "æøå". Only done once
"""with open(ordbank_lemma_path, "rb") as fil, open("ordbank_lemma.txt", "w") as write_file:
    for line in fil:
        decoded_line = line.decode("latin-1")
        write_file.write(decoded_line)

with open(ordbank_fullform_path, "rb") as fil, open("ordbank_fullform.txt", "w") as write_file:
    for line in fil:
        decoded_line = line.decode("latin-1")
        write_file.write(decoded_line)"""

# Reading new files
ordbank_lemma_path = "ordbank_lemma.txt"
ordbank_fullform_path = "ordbank_fullform.txt"
lemma_dict = {}

# making dict and json output file of ordbank lemma
with open(ordbank_lemma_path, 'r', encoding='utf-8') as fil:
    for line in fil:
        if "LOEPENR" not in line: # skip first line
            lemma_dict[line.split()[2]] = {"idx": line.split()[0], "lemma_id":line.split()[1]}
        
with open("ordbank_lemma.json", "w", encoding="utf-8") as outfile:
    json.dump(lemma_dict, outfile, ensure_ascii=False)


# making dict and json file out of ordbank fullforms
fullform_dict = {}

with open(ordbank_fullform_path, 'r', encoding='utf-8') as fil:
    for line in fil:
        if "LOEPENR" not in line: # skip first line
            fullform_dict[line.split()[2]] = {"idx": line.split()[0], "lemma_id":line.split()[1], "pos":line.split()[3], "morph_analysis":[]}
            
            for word in line.split()[4:]:
                fullform_dict[line.split()[2]]["morph_analysis"].append(word)
                if word == "normert": # stop when new category starts
                    break

with open("ordbank_fullform.json", "w", encoding="utf-8") as outfile:
    json.dump(fullform_dict, outfile, ensure_ascii=False)

