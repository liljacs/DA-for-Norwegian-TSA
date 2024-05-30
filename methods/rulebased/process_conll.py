# Makes an updated training file on conll-format with augmented data
# Paths must be changed if reproduced

import stanza
from tqdm import tqdm
import torch
from conllu import parse, parse_incr
from conllu.models import TokenList

try:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print('Apple GPU')
    else:
        device = torch.device('cpu')
except:
    device = torch.device('cpu')

# Convert the TSA conll data to DatasetDict.
def parse_conll(raw: str, sep="\t"):
    """Parses the norec-fine conll files with tab separator and sentence id"""
    
    doc_parsed = []  # One dict per sentence. meta, tokens and tags
    for sent in raw.strip().split("\n\n"):
        meta = ""
        tokens, tags = [], []
        for line in sent.split("\n"):
            if line.startswith("#") and "=" in line:
                meta = line.split("=")[-1]
            else:
                elems = line.strip().split(sep)
                assert len(elems) == 2
                tokens.append(elems[0])
                tags.append(elems[1])
        assert len(meta) > 0
        doc_parsed.append({"idx": meta, "tokens": tokens, "tsa_tags": tags })

    return doc_parsed

train_path = "tsa_conll/train.conll"
nlp = stanza.Pipeline('nb')

with open(train_path, encoding="utf8") as train:
    train = train.read()
train_parsed = parse_conll(train)

sents = []

for doc in train_parsed:
    sents.append(doc["tokens"])

#stanza.download('nb')

# adding lemma, pos and other morphological features (case)
with open("train_updated.conll", "w", encoding="utf-8") as output:
    for doc_parsed in tqdm(train_parsed):
            # konverter til streng
            sent = " ".join(doc_parsed["tokens"])
            doc = nlp(sent)

            # lemmatisering og POS-tagging
            doc_parsed["lemmas"] = [token.lemma for sent in doc.sentences for token in sent.words]
            doc_parsed["pos_tags"] = [token.upos for sent in doc.sentences for token in sent.words]
            doc_parsed["feats"] = [token.feats if token.feats else "_" for sent in doc.sentences for token in sent.words]
            doc_parsed["substituted"] = ["FALSE" for sent in doc.sentences for token in sent.words]
            output.write(f"#sent_id={doc_parsed['idx']}\n")
            output.write(f"#text={sent}\n")
            
            for i, (tokens, tags, lemmas, pos_tags, feats, substituted) in enumerate(zip(doc_parsed["tokens"],doc_parsed["tsa_tags"],doc_parsed["lemmas"],doc_parsed["pos_tags"], doc_parsed["feats"], doc_parsed["substituted"])):
                output.write(f"{tokens}\t{tags}\t{lemmas}\t{pos_tags}\t{feats}\t{substituted}\n") # removed index to match orginal
            
            output.write("\n")


data = open('output.conll2', 'r', encoding='utf-8').read()
sentences = parse(data, fields=["id", "form", "tag", "lemma", "pos"])

for sent in sentences:
    sent.metadata["text": " ".join(token["form"] for token in sent)]

#print(sentences[0][0])


# making file of BIO-tags to compare with predicted file
with open("true.txt", "w", encoding="utf-8") as output:
    with open("../modeling/augmented_data/test.conll", "r", encoding="utf-8") as test:
        test = test.read()
        sentences = parse(test, fields=["form", "tag"])
        #print(len(sentences))
        for sent in sentences:
            for token in sent:
                output.write(token["tag"]+" ")
            output.write("\n")
            
            #sent = " ".join(doc_parsed["pos_tags"])
            #doc = nlp(sent)
            #print(doc)
            # output.write("\n")


