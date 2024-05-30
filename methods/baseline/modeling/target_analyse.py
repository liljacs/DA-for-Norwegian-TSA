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

test_conll = parse_conll(open("../modeling/prompt_augmented/test.conll").read())
true = open("../modeling/true.txt").read() # true test
pred = open("../modeling/output/baseline_polarspan_2024-03-20/predictions_baseline_polarspan_seed_10.txt").read() # ser på prompt_all, pred test

true = true.split("\n")
pred = pred.split("\n")

feilanalyse = open("feilanalyse_baseline.txt", "w", encoding="utf-8")
count_correct = 0
count_wrong = 0

for true_line, pred_line, tokens in zip(true,pred,test_conll):
    if true_line.split() == pred_line.split():
        feilanalyse.write("\n\nCORRECT\n")
        feilanalyse.write(true_line)
        feilanalyse.write("\n"+" ".join(tokens["tokens"]))
        feilanalyse.write("\n")
        count_correct += 1
    else:
        feilanalyse.write("\n\nWRONG..TRUE:\n")
        for num, tag in enumerate(true_line.split()):
            feilanalyse.write(f"{num}_{tag} ")
        feilanalyse.write("\n")
        for num, token in enumerate(tokens["tokens"]):
            feilanalyse.write(f"{num}_{token} ")
        feilanalyse.write("\n")
        
        feilanalyse.write("\nPRED:\n")
        for num, tag in enumerate(pred_line.split()):
            feilanalyse.write(f"{num}_{tag} ")
        feilanalyse.write("\n")
        for num, token in enumerate(tokens["tokens"]):
            feilanalyse.write(f"{num}_{token} ")
        feilanalyse.write("\n")
        count_wrong += 1
    
feilanalyse.write(f"\nCorrect:{count_correct} Wrong:{count_wrong}")




