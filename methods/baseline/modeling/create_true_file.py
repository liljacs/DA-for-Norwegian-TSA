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

parsed_dev = parse_conll(open("prompt_augmented/dev.conll").read())
print(parsed_dev[0])

with open("true_dev.txt", "w", encoding="utf-8") as dev:
    for item in parsed_dev:
        dev.write(" ".join(item["tsa_tags"])+"\n")
