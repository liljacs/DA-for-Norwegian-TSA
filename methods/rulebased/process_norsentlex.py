# Adding morphological features to fullform norsentlex, such as case

import stanza
import json
from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

# making paths and json-files
negative_path = "Data/norsentlex/Fullform/POS_Fullform_lexicon_Negative.json"
positive_path = "Data/norsentlex/Fullform/POS_Fullform_lexicon_Positive.json"


pos_map = {"Noun":"NOUN", "Adjective":"ADJ", "Verb":"VERB", "Participle_Adjectives":"ADJ"}

# Norsentlex
fullform_negative = open(negative_path, encoding="utf-8")
fullform_positive = open(positive_path, encoding="utf-8")

negative_dict = json.load(fullform_negative)
positive_dict = json.load(fullform_positive)

new_negative_dict = {}
new_positive_dict = {}

nlp = stanza.Pipeline("nb")


# MAKING DICTIONARIES
# New negative dictionary with more tags
logging.info("TRAVERSING NEGATIVE DICTIONARY")
for key, val in tqdm(negative_dict.items()):
    for pos, tokens in val["forms"].items():
        text = " ".join(tokens)
        doc = nlp(text)
        
        tokens_dict = {}

        for sent in doc.sentences:
            for i, token in enumerate(sent.words):
                if pos_map[pos] != token.upos:
                    token.upos = pos_map[pos] 
                
                new_negative_dict[tokens[i]] = {"pos":token.upos, "feats": token.feats if token.feats else "_"}

# New positive dictionary with more tags
logging.info("TRAVERSING POSITIVE DICTIONARY")
for key, val in tqdm(positive_dict.items()):
    for pos, tokens in val["forms"].items():
        text = " ".join(tokens)
        doc = nlp(text)

        for sent in doc.sentences:
            for i, token in enumerate(sent.words):
                if pos_map[pos] != token.upos:
                    token.upos = pos_map[pos] 
                
                new_positive_dict[tokens[i]] = {"pos":token.upos, "feats": token.feats if token.feats else "_"}

json_negativ = "fullform_negative.json"
json_positiv = "fullform_positive.json"

with open(json_negativ, 'w', encoding='utf-8') as negativ, open(json_positiv, 'w', encoding='utf-8') as positiv:
    json.dump(new_negative_dict, negativ)
    json.dump(new_positive_dict, positiv)

logging.info("DONE")
