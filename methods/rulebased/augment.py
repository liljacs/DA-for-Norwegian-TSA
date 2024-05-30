# finds words present in NorSentLex
# switches the words with the most similar synonym
# ignores participle -- don't have the tag for it

from conllu import parse
from conllu.serializer import serialize
from tqdm import tqdm
import json
import os
import random
import logging

lemma_path = "Data/norsentlex-master/Lemma"
pos_map = {"ADJ":"ADJECTIVE", "NOUN":"NOUN", "VERB":"VERB"}
morph_map = {"Number=Plur":"fl", "Number=Sing":"ent", "Definite=Ind":"ub", "Definite=Def":"be", "Tense=Inf":"inf", "Tense=Pres":"pres", "Tense=Imp":"imp", "Tense=Past":"pret", "Voice=Pass":"pass", "Gender=Neut":"nøyt",\
             "Degree=Cmp":"komp", "Degree=Sup":"sup"}
file_list = [fil for fil in os.listdir(lemma_path) if "PARTICIPLE" not in fil]

# reading ordbank data and converting to dictionary
ordbank_lemma = open("Data/ordbank_lemma.json", "r", encoding="utf-8")
ordbank_fullform = open("Data/ordbank_fullform.json", "r", encoding="utf-8")

lemma_dict = json.load(ordbank_lemma)
fullform_dict = json.load(ordbank_fullform)

stopwords_file = open("Data/stopwords.txt", "r") # contains a few bad words from NorSentLex, all negative
stopwords = stopwords_file.read().split()
stopwords_file.close()

changes_dict = {}
changes_count = 0
pos_count = 0
neg_count = 0
changed_list = [None]


def substitute(random_lemma, norec_analysis):
    """Substitutes a word in a sentence with a random word from NorSentLEx, 
    if the orignial word is present in NorSentLEx. Saves the full morphological form
    of original word, and finds same inflection of the substitued word in "ordbanken".
    """
    substituted = False
    if random_lemma in lemma_dict and random_lemma != token["lemma"]: # finding lemma in ordbanken
        lemma_id = lemma_dict[random_lemma]["lemma_id"] # finding lemma id to use in fullform dict
        for fullform, info in fullform_dict.items():
            if info["lemma_id"] == lemma_id:
                relevant_feats = []
                # find same inflection 
                for feat in norec_analysis:
                    if feat in morph_map: # ignore some of the features, only use those in map
                        relevant_feats.append(feat)

                if all(morph_map[i] in info["morph_analysis"] for i in relevant_feats) and len(relevant_feats)>0:
                    if token["form"] == changed_list[-1]: # skip if word already substituted 
                        continue
                    changes_dict["old_lemma"] = token["lemma"]
                    changes_dict["old_fullform"] = token["form"]
                    changes_dict["new_lemma"] = random_lemma
                    changes_dict["new_fullform"] = fullform
                    changes_dict["id"] = token["id"]
                    
                    token["lemma"] = random_lemma # substitute random word with same sentiment
                    token["form"] = fullform
                    token["substituted"] = "TRUE"
                    changed_list.append(fullform)
                    substituted = True

        return substituted


with open("Data/train_updated.conll", "r", encoding="utf-8") as fil:
    fil = fil.read()
    sentences = parse(fil, fields=["id", "form", "tag", "lemma", "pos", "morph_analysis", "substituted"])
    
    logging.info("TRAVERSING SENTENCES")
    for sent in tqdm(sentences):
        is_negative = any(token["tag"] == "B-targ-Negative" or token["tag"] == "I-targ-Negative" for token in sent)
        is_positive = any(token["tag"] == "B-targ-Positive" or token["tag"] == "I-targ-Positive" for token in sent)

        # if sentence is negative, check negative lexicon. 
        if is_negative:
            for token in sent:
                for filename in file_list:
                    if "Negative" in filename:
                        data = open(lemma_path+"/"+filename)
                        lexicon = data.read().split()

                        if token["pos"] in pos_map and token["lemma"] in lexicon and pos_map[token["pos"]] in filename:
                            norec_analysis = token["morph_analysis"].split("|")
                            random_lemma = random.choice(lexicon)

                            while random_lemma in stopwords: # choose new word if in stopwords
                                random_lemma = random.choice(lexicon)

                            result = substitute(random_lemma, norec_analysis)
                            # try different random word if not present in ordbanken
                            if result is False:
                                random_lemma = random.choice(lexicon)
                                substitute(random_lemma, norec_analysis)
                            else:
                                changes_count += 1 # keeps track of number of substitutions
                                neg_count += 1


        # if sentence is positive, check positive lexicon
        elif is_positive:
            for token in sent:
                for filename in file_list:
                    if "Positive" in filename:
                        data = open(lemma_path+"/"+filename)
                        lexicon = data.read().split()

                        if token["pos"] in pos_map and token["lemma"] in lexicon and pos_map[token["pos"]] in filename:
                            norec_analysis = token["morph_analysis"].split("|")
                            random_lemma = random.choice(lexicon)
                            
                            result = substitute(random_lemma, norec_analysis)
                            # try different random word if not present in ordbanken
                            if result is False:
                                random_lemma = random.choice(lexicon)
                                substitute(random_lemma, norec_analysis)
                            else:
                                changes_count += 1 # keeps track of number of substitutions
                                pos_count += 1
        
        # re-writing conllu file
        sentence_data = serialize(sent)
        
        with open("Data/augmented_train.conllu", "a", encoding="utf-8") as f:
            f.write(sentence_data)

print("All changes:", changes_count)
print("negative:", neg_count)
print("positive:", pos_count)

logging.info("DONE")

