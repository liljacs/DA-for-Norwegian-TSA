import json
import re
import nltk
import random

#nltk.download('punkt')

def make_conll(train_path, json_path):
    with open(train_path, "w", encoding="utf-8") as augment_file:
        fila = open(json_path)
        prompt_dict = json.load(fila)
        fila.close()

        # for each sentence/response
        augment_count = 0
        for num_sent in range(0, len(prompt_dict.keys())-1):
            values = prompt_dict[str(num_sent)]
            response = values["response"]
            gold_target = values["target"]
            polarity = values["polarity"]
            target_info = []
            pattern = r'{}(?:\w{{1,3}}\b)?'.format(gold_target.lower()) # allow case after target, and eiendoms-s
            ai_pattern = r'(?:[Ss]om (AI-?)?\s?(språk\s?modell))' # we dont want to match these
            matching_targets = re.findall(pattern=pattern, string=response.lower())
            split_response = nltk.word_tokenize(response.lower())
            not_lowered_response = nltk.word_tokenize(response) # to write in file
            indices = [] # indices to find position of target
            
            if len(split_response) > 1 and not re.search(ai_pattern, response):
                re.sub(r'"(.*?)"', r'\1', response) # removes quotations that surround whole response
                if len(gold_target.split()) > 1: # if target has more than one token
                    B_targ = gold_target.split()[0] 
                    
                    for num, word in enumerate(split_response):
                        if word.lower() == B_targ.lower():
                            if " ".join(split_response[num:num+len(gold_target.split())]) in matching_targets:
                                indices.append((num, num+len(gold_target.split())-1)) #start, end
                    
                else: # target is one token only
                    for num, word in enumerate(split_response):
                        if word.lower() in matching_targets and len(matching_targets) < 4: # many matches can mean repetetive response
                            indices.append((num, num)) # start, end
                if len(indices) > 0: # empty if no matches
                    augment_file.write(f"#sent_id=augmented_{augment_count}\n")
                    bio_tags = ["O" for i in range(len(split_response))]
                    for pair in indices:
                        if polarity == "Positiv":
                            target_info.append((pair[0], pair[1], "Positive"))
                        elif polarity == "Negativ":
                            target_info.append((pair[0], pair[1], "Negative"))

                        start = target_info[0][0]
                        end = target_info[0][1]
                        targ_polarity = target_info[0][2]

                        bio_tags[start] = f"B-targ-{targ_polarity}" # kan være flere targs i en setning

                        if end != start:
                            for i in range(start+1,end+1):
                                bio_tags[i] = f"I-targ-{targ_polarity}"
                        
                        assert len(bio_tags) == len(split_response)
                        

                    for token,tag in zip(not_lowered_response, bio_tags):
                        augment_file.write(f"{token}\t{tag}\n")    
                    
                    # new sentence
                    augment_count += 1
                    augment_file.write("\n")
        print("augmented sents:", augment_count)
            

def parse_conll(raw:str, sep="\t"):
    """Parses the norec-fine conll files with tab separator and sntence id"""
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


def clean_quotes(path):
    """The file contains a set of quotations which Python does not recognize unless one makes the
    file bianry first. Removes these quotations."""
    count_quotes = 0

    fil = open(path, "rb").read()
    content_str = fil.decode('utf-8', errors='ignore')
    content_str = content_str.replace("``", '"')
    content_str = content_str.replace("''", '"')
    content_modified = content_str.encode('utf-8', errors='ignore')

    with open(path, 'wb') as modified_file:
        modified_file.write(content_modified)
    
    new_modified = open(path, "r", encoding="utf-8").read()
    parsed_modified = parse_conll(new_modified)

    with open(path, "w", encoding="utf-8") as final_out:
        for sent in parsed_modified:
            response = sent["tokens"]
            if sent["tsa_tags"][0] == "O" and sent["tsa_tags"][-1] == "O":
                if re.match(r'^"(.*?)"$', " ".join(response)): 
                    count_quotes += 1
                    response = response[1:-1] # removes quotations that surround whole response
                    sent["tsa_tags"] = sent["tsa_tags"][1:-1]

            final_out.write(f"#sent_id={sent['idx']}\n")
                
            for token, tag in zip(response, sent["tsa_tags"]):
                final_out.write(f"{token}\t{tag}\n")
            final_out.write("\n")
        
    print("cleaned quotes:", count_quotes)
    print(len(open(path).read().split("\n\n")))


def concatenate_files(original, augmented, only=False):
    """Concatenates the origianl train file with the augmented."""
    with open(original, "r", encoding="utf-8") as original_file, open(augmented, "r", encoding="utf-8") as augmented_file:
        original_file = original_file.read()
        augmented_file = augmented_file.read()

    if only:
        filename = augmented[:24]
    
    with open(filename, "w", encoding="utf-8") as concat:
        concat.write(original_file+"\n")
        concat.write(augmented_file)
    
    parsed_augmented = parse_conll(open(filename, "r", encoding="utf-8").read())
    random.shuffle(parsed_augmented)
    
    with open(filename, "w", encoding="utf-8") as final_out:
        for sent in parsed_augmented:
            final_out.write(f"#sent_id={sent['idx']}\n")
            
            for token, tag in zip(sent["tokens"], sent["tsa_tags"]):
                final_out.write(f"{token}\t{tag}\n")
            
            final_out.write("\n")
    
    print(len(open(filename).read().split("\n\n")))


def concatenate_all(file1, file2, file3, original_file=None, only=False):
    """Concatenates to one bigh training dataset."""
    if original_file:
        original = open(original_file, "r", encoding="utf-8")
    
    if only:
        filename = "augmented/train_prompt_all_only.conll"
    else:
        filename = "augmented/train_prompt_all.conll"

    with open(file1, "r", encoding="utf-8") as prompt1, open(file2, "r", encoding="utf-8") as prompt2, \
         open(file3, "r", encoding="utf-8") as prompt3:
        
        if original_file:
            original = original.read()
        prompt1 = prompt1.read()
        prompt2 = prompt2.read()
        prompt3 = prompt3.read()
        
    with open(filename, "w", encoding="utf-8") as concat:
        if original_file:
            concat.write(original)
        concat.write(prompt1)
        concat.write(prompt2)
        concat.write(prompt3)
    
    parsed_augmented = parse_conll(open(filename).read())
    random.shuffle(parsed_augmented)

    with open(filename, "w", encoding="utf-8") as final_out: # change file name if needed
        for sent in parsed_augmented:
            final_out.write(f"#sent_id={sent['idx']}\n")
            
            for token, tag in zip(sent["tokens"], sent["tsa_tags"]):
                final_out.write(f"{token}\t{tag}\n")
            
            final_out.write("\n")
    
    print(len(open(filename).read().split("\n\n")))


def double_dataset(old_path, new_path):
    """Doubles one of the datasets. Can be used to check if amount of data is important."""
    with open(old_path, "r", encoding="utf-8") as old:
        content = old.read()

        with open(new_path, "w", encoding="utf-8") as new:
            new.write(content)
            new.write(content)
    
    parsed_augmented = parse_conll(open(new_path, "r", encoding="utf-8").read())
    random.shuffle(parsed_augmented)
    print(len(open(new_path).read().split("\n\n")))
    

if __name__ == "__main__":

    """make_conll("augmented/train_prompt_1_only.conll", "augmented/train_prompt_1.json")
    clean_quotes("augmented/train_prompt_1_only.conll")
    # prompt 2 only
    make_conll("augmented/train_prompt_2_only.conll", "augmented/train_prompt_2.json")
    clean_quotes("augmented/train_prompt_2_only.conll")
    # prompt 3 only
    make_conll("augmented/train_prompt_3_only.conll", "augmented/train_prompt_3.json")
    clean_quotes("augmented/train_prompt_3_only.conll")
    # concatenation
    concatenate_files("norec_data/train.conll", "augmented/train_prompt_1_only.conll", only=True)
    # concatenate prompt 2
    concatenate_files("norec_data/train.conll", "augmented/train_prompt_2_only.conll", only=True)
    # concatenate prompt 3
    concatenate_files("norec_data/train.conll", "augmented/train_prompt_3_only.conll", only=True)"""
    # concatenate all prompts
    #concatenate_all("norec_data/train.conll", "augmented/train_prompt_1_only.conll","augmented/train_prompt_2_only.conll", "augmented/train_prompt_3_only.conll", only=False)
    # concatenate all prompts without norec
    #concatenate_all(file1="augmented/train_prompt_1_only.conll",file2="augmented/train_prompt_2_only.conll",file3="augmented/train_prompt_3_only.conll", only=True)
    # doubling the all-dataset with norec
    #double_dataset("augmented/train_prompt_all.conll", "augmented/train_prompt_all_double.conll")
