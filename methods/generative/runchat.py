from chat import Chat
from norec_helpers import parse_conll
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import re
from random import sample, choice
import os
import csv
import textwrap
import pandas as pd
import stanza
import json

# loading norec data
parsed_train = parse_conll(open("norec_data/train.conll", "r", encoding="utf-8").read())
parsed_dev = parse_conll(open("norec_data/dev.conll", "r", encoding="utf-8").read())

# subset
train_subset = sample(parsed_train, 50) # random samples from train subset, high number to ensure at least 20 targets
subset_IOB = [sent["tsa_tags"] for sent in train_subset]
subset_tokens = [sent["tokens"] for sent in train_subset]
subset_targets = []

# dev data for few-shot
dev_tokens = [" ".join(sent["tokens"]) for sent in parsed_dev]
dev_tags = [sent["tsa_tags"] for sent in parsed_dev]
dev_neg = []
dev_pos = []
dev_mixed = [] 

# sort positive, negative and mixed sentences from dev, for few-shot
for tokens, tags in zip(dev_tokens, dev_tags):
    if "B-targ-Positive" in tags and "B-targ-Negative" not in tags:
        dev_pos.append(tokens)
    elif "B-targ-Negative" in tags and "B-targ-Positive" not in tags:
        dev_neg.append(tokens)
    elif "B-targ-Negative" in tags and "B-targ-Positive" in tags:
        dev_mixed.append(tokens)

beginning = "B-"
inside = "I-"

while len(subset_targets) <= 20: # generate 30 to make sure that there are at least 20 unique targets
    for tokens, tags in zip(subset_tokens, subset_IOB): 
        num_tag = 0 # reset when new sentence
        for num_tag, (token, tag) in enumerate(zip(tokens, tags)):
            if beginning in tags[num_tag]:
                sublist = []
                sublist.append(tokens[num_tag]) # appending token corresponding to B-tag 
                if num_tag == len(tokens)-1: # dont check rest if there are no more tokens
                    subset_targets.append((*sublist, tags[num_tag][7:14]))
                else:
                    for i in range(num_tag+1, len(tokens)): # cheking rest of sequence
                        if inside in tags[i]:
                            sublist.append(tokens[i])
                        elif tags[i] == "O": # if "O" is current tag, last target is finished
                            if len(sublist) > 1:
                                subset_targets.append((" ".join(sublist), tags[i-1][7:14]))
                            elif len(sublist) == 1:
                                subset_targets.append((*sublist, tags[i-1][7:14]))
                            sublist = [] # reset list if tag is "O"
                            num_tag = i
                            break
                        elif beginning in tags[i]: # if we find new "B", start new target
                            num_tag = i  
                            break                     

subset_targets = subset_targets[:20] # because the amount of targets within a sentence can't be controlled, we limit the final size 

def find_all_targets():
    # full train data, for main augmentation
    all_IOB = [sent["tsa_tags"] for sent in parsed_train]
    all_tokens = [sent["tokens"] for sent in parsed_train]
    all_targets = []

    for tokens, tags in zip(all_tokens, all_IOB): 
        num_tag = 0 # reset when new sentence
        for num_tag, (token, tag) in enumerate(zip(tokens, tags)):
            if beginning in tags[num_tag]:
                sublist = []
                sublist.append(tokens[num_tag]) # appending token corresponding to B-tag 
                if num_tag == len(tokens)-1: # dont check rest if there are no more tokens
                    all_targets.append((*sublist, tags[num_tag][7:14]))
                else: # check rest of sequence
                    for i in range(num_tag+1, len(tokens)):
                        if inside in tags[i]:
                            sublist.append(tokens[i])
                            if i == len(tokens)-1:
                                all_targets.append((" ".join(sublist), tags[i-1][7:14])) # If "I" is last tag in sentence
                                break
                        elif tags[i] == "O": # if "O" is current tag, last target is finished
                            if len(sublist) > 1:
                                all_targets.append((" ".join(sublist), tags[i-1][7:14]))
                            elif len(sublist) == 1:
                                all_targets.append((*sublist, tags[i-1][7:14]))
                            sublist = [] # reset list if tag is "O"
                            num_tag = i
                            break
                        elif beginning in tags[i]: # if we find new "B", start new target
                            if len(sublist) > 1:
                                all_targets.append((" ".join(sublist), tags[i-1][7:14]))
                            elif len(sublist) == 1:
                                all_targets.append((*sublist, tags[i-1][7:14]))
                            num_tag = i  
                            break 
    
    return all_targets, all_tokens

chat = Chat()

def insert_newlines(val):
    """Help method to insert newlines when a line is long."""
    return '\n'.join(textwrap.wrap(str(val), width=60))

nlp = stanza.Pipeline(lang='nb', processors='tokenize')

def chat_zero_shot(prompt_id:any, prompt:str):
    """Writes the output zero-shot prompts using ChatNorT5 to files."""

    prompt_dict = {"Target":[], "Polarity":[], f"Prompt_ID:{prompt_id}, Prompt: {prompt}":[], "target":[], "polaritet":None, "fluency":None, "én setning":[], "sammenheng":None, "AI-nekt":None, "HTML-symbol":[]}
    
    for (targ, polar), tokens in tqdm(zip(subset_targets, subset_tokens)):
        if prompt_id[-1] == "2":
            if polar =="Positiv":
                polar = "bra"
            else:
                polar = "dårlig"

        modified_prompt = prompt.format(target=targ, polarity=polar.lower()) # fjerner +t når verbet er 'setning'    
        prompt_dict["Target"].append(f"{targ}")
        prompt_dict["Polarity"].append(f"{polar}")
        svar = chat(modified_prompt)
        prompt_dict[f"Prompt_ID:{prompt_id}, Prompt: {prompt}"].append(svar)
        
        doc = nlp(svar)
        if len(doc.sentences) > 1:
            prompt_dict["én setning"].append(1)
        else:
            prompt_dict["én setning"].append(0)
        
        if targ.lower().strip('"') in svar.lower().strip('"'): # OBS: fjernet split. Sjekk om det gir mange feil, f.eks. at target gjenkjennes som del av lengre ord, f.eks. telefon = telefonnummer
            prompt_dict["target"].append(0)
        else:
            prompt_dict["target"].append(1)
        
        if "<br>" in svar:
            prompt_dict["HTML-symbol"].append(1)
        else:
            prompt_dict["HTML-symbol"].append(0)

    
    prompt_df = pd.DataFrame(prompt_dict)
    prompt_df = prompt_df.applymap(insert_newlines) # inserting newlines
    prompt_df.to_excel(f"prompt_bra_vs_pos_neg/{prompt_id}.xlsx", encoding="utf-8")

prompts = [
    # med anførselstegn
    #{"id":"1-1-1-1-1", "prompt": "Skriv noe {polarity} om '{target}'."},
    #{"id":"2-1-1-1-1", "prompt":"'{target}' er {polarity}. Skriv noe {polarity} om '{target}'."}, # må kjøre denne på nytt forid fant feil
    #{"id":"3-1-1-1-1", "prompt":"Du er en anmelder. Skriv noe {polarity} om '{target}'."},
    #{"id":"4-1-1-1-1", "prompt":"Lat som du er en anmelder. Skriv noe {polarity} om '{target}'."},

    # uten anførselstegn, kjører på nytt fordi glemte å fjerne noen anførselstegn
    #{"id":"1-2-1-1-1", "prompt": "Skriv noe {polarity} om {target}."},
    #{"id":"2-2-1-1-1", "prompt":"{target} er {polarity}. Skriv noe {polarity} om {target}."}, # må kjøre denne på nytt forid fant feil
    #{"id":"3-2-1-1-1", "prompt":"Du er en anmelder. Skriv noe {polarity} om {target}."},
    #{"id":"4-2-1-1-1", "prompt":"Lat som du er en anmelder. Skriv noe {polarity} om {target}."},

    # ny andre runde. grunnprompt 1 med setning og noe.
    #{"id":"1-1-1-1-1", "prompt": "Skriv noe {polarity} om '{target}'."},
    #{"id":"1-1-1-2-1", "prompt": "Skriv en {polarity} setning om '{target}'."},

    
    # tredje runde
    # med anførselstegn, "skriv", "setning", grunnprompt 1 
    #{"id":"1-1-1-2-2", "prompt": "Skriv en {polarity} setning om '{target}'."},
    # med anførselstegn, "si", "setning", grunnprompt 1  
    #{"id":"1-1-2-2-2", "prompt": "Si en {polarity} setning om '{target}'."},
    # med anførselstegn, "generer", "setning", grunnprompt 1  
    #{"id":"1-1-3-2-2", "prompt": "Generer en {polarity} setning om '{target}'."},  
    # med anførselstegn, "skriv", "setning", grunnprompt 1 
    {"id":"1-1-1-2-1", "prompt": "Skriv en {polarity} setning om '{target}'."},
    # med anførselstegn, "si", "setning", grunnprompt 1  
    {"id":"1-1-2-2-1", "prompt": "Si en {polarity} setning om '{target}'."},
    # med anførselstegn, "generer", "setning", grunnprompt 1  
    {"id":"1-1-3-2-1", "prompt": "Generer en {polarity} setning om '{target}'."},   
]
        

def chat_few_shot(prompt_id:any, prompt:str):
    """Writes the output zero-shot prompts using ChatNorT5 to files."""

    prompt_dict = {"Target":[], "Polarity":[], "shots":[], f"Prompt_ID:{prompt_id}, Prompt: {prompt}":[], "target":[], "polaritet":None, "fluency":None, "én setning":[], "sammenheng":None, "AI-nekt":None, "HTML-symbol":[]}
    
    for (targ, polar), tokens in tqdm(zip(subset_targets, subset_tokens)):
        modified_prompt = prompt.format(target=targ, polarity=polar.lower()+"t") # fjerner +t når verbet er 'setning'
        positive_prompt = prompt.format(target=targ, polarity="positivt") # for first part of few shot
        negative_prompt = prompt.format(target=targ, polarity="negativt") # for second part of few shot
        
        two_shot = [choice(dev_pos), choice(dev_neg)] # ['str', 'str']
        four_shot = [sample(dev_pos, 2), sample(dev_neg, 2)] # [['str', 'str'], ['str', 'str']]

        prompt_dict["Target"].append(f"{targ}")
        prompt_dict["Polarity"].append(f"{polar}")
        svar = chat(modified_prompt, few_shots=[[positive_prompt, four_shot[0][0]], 
                                                [positive_prompt, four_shot[0][1]],
                                                [negative_prompt, four_shot[1][0]],
                                                [negative_prompt, four_shot[1][1]]])
        prompt_dict[f"Prompt_ID:{prompt_id}, Prompt: {prompt}"].append(svar)
        
        doc = nlp(svar)
        if len(doc.sentences) > 1:
            prompt_dict["én setning"].append(1)
        else:
            prompt_dict["én setning"].append(0)
        
        if targ.lower().strip('"') in svar.lower().strip('"'): # OBS: fjernet split. Sjekk om det gir mange feil, f.eks. at target gjenkjennes som del av lengre ord, f.eks. telefon = telefonnummer
            prompt_dict["target"].append(0)
        else:
            prompt_dict["target"].append(1)
        
        if "<br>" in svar:
            prompt_dict["HTML-symbol"].append(1)
        else:
            prompt_dict["HTML-symbol"].append(0)

        prompt_dict["shots"].append(f"POS: {four_shot[0][0],four_shot[0][1]}, NEG: {four_shot[1][0], four_shot[1][1]}")
    
    prompt_df = pd.DataFrame(prompt_dict)
    prompt_df = prompt_df.applymap(insert_newlines) # inserting newlines
    prompt_df.to_excel(f"4-shot/{prompt_id}.xlsx", encoding="utf-8")


for prompt in prompts:
    chat_few_shot(prompt["id"], prompt["prompt"])

# zero shot
"""for prompt in prompts:
    chat_zero_shot(prompt["id"], prompt["prompt"])"""

prompt_dict = {} 

def augment_data(prompt_id:any, prompt:str):
    """Augments the Norec data using ChatNorT5"""
    all_targets, all_tokens = find_all_targets() # 5044 av 5044 targets i TSA (6778 i norec fine)
    prompt_dict["prompt_id"] = prompt_id
    added = 0
    not_sentence = 0
    not_target = 0

    for (targ, polar), tokens in tqdm(zip(all_targets, all_tokens)):
        modified_prompt = prompt.format(target=targ, polarity=polar.lower())
        svar = chat(modified_prompt)
        
        doc = nlp(svar)
        if len(doc.sentences) == 1:
            if targ.lower().strip('"') in svar.lower().strip('"'): # OBS: fjernet split. Sjekk om det gir mange feil, f.eks. at target gjenkjennes som del av lengre ord, f.eks. telefon = telefonnummer
                prompt_dict[added] = {"response": svar, "target":targ, "polarity":polar}
                added = added + 1
            else:
                not_target = not_target + 1
        else:
            not_sentence = not_sentence + 1
    
    with open("augmented/train_prompt_3.json", "w", encoding="utf-8") as out:
        json.dump(prompt_dict, out, ensure_ascii=False)

    print("number of augmented sents:", added)
    print("non-valid sentences:", not_sentence)
    print("non-valid targets:", not_target)


"""
for prompt in prompts:
    augment_data(prompt["id"], prompt["prompt"])"""




                    








