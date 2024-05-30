
def parse_conll(raw:str, sep="\t"):
    """Parses the norec conll files with tab separator and sntence id"""
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