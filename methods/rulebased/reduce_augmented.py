from conllu import parse
from conllu.serializer import serialize
from tqdm import tqdm
import json
import os
import random
import logging
from collections import Counter
from parse_conllu import parse_conll

#changed_count = 0
#unchanged_count = 0

# reduce file to only contain changed sentences
with open("augmented_train.conllu", "r", encoding="utf-8") as file:
    fil = fil.read()
    sentences = parse(fil, fields=["id", "form", "tag", "lemma", "pos", "morph_analysis", "substituted"])
    
    logging.info("TRAVERSING SENTENCES")
    for sent in tqdm(sentences):
        if any(token["substituted"] == "TRUE" for token in sent):
            changed_count += 1
            sentence_data = serialize(sent)
            # write to file if the sentence is changed
            with open("augmented_train_clean.conll", "a", encoding="utf-8") as f:
                f.write(sentence_data)
        
        else:
            unchanged_count += 1

print(f"Number of changed sentences: {changed_count}")
print(f"Number of unchanged sentences: {unchanged_count}")
print(f"Total: {changed_count+unchanged_count}")

# count number of changed tokens
changed_token_count = 0
unchanged_token_count = 0
positive_count = 0
negative_count = 0

pos_positive = []
pos_negative = []
pos_unchanged = []

with open("augmented_train_clean.conll", "r", encoding="utf-8") as fil:
    fil = fil.read()
    sentences = parse(fil, fields=["id", "form", "tag", "lemma", "pos", "morph_analysis", "substituted"])

    with open("Data/norsentlex-master/Fullform/POS_Fullform_lexicon_Negative.json", "r", encoding="utf-8") as negative:
        negative = json.load(negative)
    
    with open("Data/norsentlex-master/Fullform/POS_Fullform_lexicon_Positive.json", "r", encoding="utf-8") as positive:
        positive = json.load(positive)
    
    logging.info("TRAVERSING SENTENCES")
    for sent in tqdm(sentences):
        for token in sent:
            if token["substituted"] == "TRUE":
                changed_token_count += 1
                if token["lemma"] in positive.keys():
                    positive_count += 1
                    pos_positive.append(token["pos"])
                
                elif token["lemma"] in negative.keys():
                    negative_count += 1
                    pos_negative.append(token["pos"])
            else:
                pos_unchanged.append(token["pos"])
                unchanged_token_count += 1

print(f"changed tokens: {int(changed_token_count)}")
print(f"unchanged tokens: {unchanged_token_count}")
print(f"changed positive tokens: {positive_count}")
print(f"changed negative tokens: {negative_count}+\n")
counter_positive = Counter(pos_positive)
counter_negative = Counter(pos_negative)
unchanged_counter = Counter(pos_unchanged)


print("positive changes:", counter_positive,"\n")
print("negative changes:", counter_negative, "\n")
print("unchanged tokens:", unchanged_counter)

# Concatenate original and augmented data
# Starting with 20%, then 40, 50 etc.
with open("augmented_train_clean.conll", "r", encoding="utf-8") as fil:
    fil = fil.read()
    sizes = [0.2, 0.4, 0.6, 0.8, 1]
    
    for size in sizes:
        augmented_sents = parse(fil, fields=["id", "form", "tag"]) # reducing tags to match original
        augmented_sents = augmented_sents[:int(len(augmented_sents)*size)+1]

        with open(f"train_updated.conll", "r", encoding="utf-8") as original_file, open(f"../modeling/augmented_data/train_augmented_{size}.conll", "a", encoding="utf-8") as augmented_file:
            original_sents = parse(original_file.read(), fields=["id", "form", "tag"])
            
            for sent in original_sents:
                sent.metadata.pop("text", None) # remove key to match original
            for sent in augmented_sents:
                sent.metadata.pop("text", None)

            for sent in original_sents:
                augmented_file.write(serialize(sent)) 
            
            for sent in augmented_sents:
                augmented_file.write(serialize(sent))

        parsed_augmented = parse_conll(open(f"../modeling/augmented_data/train_augmented_{size}.conll", "r", encoding="utf-8").read())
        random.shuffle(parsed_augmented)
        
        # rewrite shuffled data
        """with open(f"../modeling/augmented_data/train_augmented_{size}.conll", "w", encoding="utf-8") as final_out:
            for sent in parsed_augmented:
                final_out.write(f"#sent_id={sent['idx']}\n")
                
                for token, tag in zip(sent["tokens"], sent["tsa_tags"]):
                    final_out.write(f"{token}\t{tag}\n")
                
                final_out.write("\n")"""
            




