from sklearn.metrics import f1_score
from itertools import product
import re

class Analysis:
    def __init__(self):
        self.adjacent_chunks = 0 # chunks next to each other
        self.broken_chunks = 0 # chunks starting with "I-targ"
        self.mixed_chunks = 0 # chunks with mixed polarity
        self.threshold = 0.5
        self.totalchunks = 0 # number of cases where both have at least one chunk


    def readfiles(self, predpath, goldpath):
        """Reads the predicted and gold file. Checks if the y have same length.
        If not, it removes potential extra lineshifts."""
        print("pred:", predpath, "gold:", goldpath)
        with open(predpath,"r",encoding="utf-8") as indata:
            self.pred = indata.read().split("\n")

        with open(goldpath,"r",encoding="utf-8") as indata:
            self.true = indata.read().split("\n")
        
        # fix some cases with an extra tag
        changed_line = False
        with open(predpath, "w", encoding="utf-8") as new_pred_file:
            for num, (predline, goldline) in enumerate(zip(self.pred, self.true)):
                #if num == 1045 and len(predline.split()) != len(goldline.split()):
                changed_line = True
                if len(predline.split()) != len(goldline.split()) and len(predline.split()) != 0 and len(goldline.split()) != 0:
                    predline = predline.strip()[:-1]   
                    new_pred_file.write(predline+"\n")
                else:
                    new_pred_file.write(predline+"\n") 
        
        if changed_line:
            with open(predpath, "r", encoding="utf-8") as pred:
                self.pred = pred.read().split("\n")

        predlen = len(self.pred)
        goldlen = len(self.true)

        if predlen != goldlen:
            print("WARNING: PRED LENGTH NOT SAME AS GOLD")
        #print(self.pred[-1] == "") # if extra lineshift
        if self.pred[-1] == "":
            print("Automatically fixing extra lineshift")
            self.pred = self.pred[:-1] # removing extra line
            predlen = len(self.pred)
        
        print("Length after fix:")
        print("Pred length,", predlen)
        print("Gold length,", goldlen)

    def chunks(self, lista):
        """Takes a list (a sequence) as input, and finds the BIO chunks in them."""
        
        lista = lista.split(" ")
        chunksa = []
        newchunk = []
        listiter = iter(lista)
        current = next(listiter,"|||")
        
        while current != "|||":
            if current == "O":  # finishes a chunk
                if len(newchunk) != 0: 
                    chunksa.append(newchunk)
                    newchunk = [] # reset list when chunk is finished
            elif "B-targ" in current:
                if len(newchunk) != 0: # there are two chunks next to each other
                    chunksa.append(newchunk)
                    newchunk = [current] # Start new chunk with B-targ
                    self.adjacent_chunks += 1
                else:
                    newchunk.append(current) # Start new chunk
            else:
                newchunk.append(current) # add I-targ to chunk
            current = next(listiter,"|||")
        return chunksa


    def numchunks(self, lista):
        """Same as chunks(), but additionally keeps the indices.
        Each element is a tuple. """       
        
        lista = lista.split(" ")
        chunksa = []
        newchunk = []
        listiter = iter(lista)
        index = 0
        current = next(listiter,"|||")

        while current != "|||":
            if current == "O": # finishes a chunk
                if len(newchunk) != 0: 
                    chunksa.append(newchunk)
                    newchunk = []
            elif "B-targ" in current:
                if len(newchunk) != 0: # there are two chunks next to each other
                    chunksa.append(newchunk)
                    newchunk = [(current,index)] # Start new chunk with B-targ
                    self.adjacent_chunks += 1
                else:
                    newchunk.append((current,index)) # Start new chunk
            else:
                newchunk.append((current,index)) # add I-targ to the chunk
            current = next(listiter,"|||")
            index += 1
        return chunksa


    def consistency(self, chunklist):
        """Checks if a list of chunks is consistent in terms of polarity"""
        
        for chunk in chunklist:
            if "I-targ" in chunk[0]:
                self.broken_chunks += 1 # chunk starts with "I-targ"
            haspos = any(["Positive" in tag for tag in chunk])
            hasneg = any(["Negative" in tag for tag in chunk])
            
            if haspos and hasneg:
                #print(chunklist)
                self.mixed_chunks += 1
    

    def cons_small(self, chunk):
        """Checks consistency in terms of polarity in tuples"""

        if chunk in [['B-targ-Negative', 'I-targ-Positive'],
                     ['B-targ-Positive', 'I-targ-Negative'],
                     ['B-targ-Negative', 'I-targ-Positive']]:
            print("CHUNK",chunk)
        #checks if a single chunk is consistent
        haspos = any(["Positive" in tags[0] for tags in chunk])
        hasneg = any(["Negative" in tags[0] for tags in chunk])
        if haspos and hasneg:
            print("Invalid chunk")
            
            return False
        return True


    def compare(self, chunks1, chunks2):
        """
        Compares two consistent sentences.
        Returns: lists for pred and gold, consisting of polarity tags if the sentences are
        evaluative.
        """

        pred_result = []
        gold_result = []
        chunks1 = [chunk for chunk in chunks1 if self.cons_small(chunk)]
        chunks2 = [chunk for chunk in chunks2 if self.cons_small(chunk)]
        prod = product(chunks1, chunks2) # all possible pairs, because we dont know which chunk corresponds to which
        
        for pair in prod:
            # for every possible pair, check if there is 50% similarity or more. 
            # if it is, it is likely that the chunks were 'intended' to represent the same words

            pred_num = set([p[1] for p in pair[0]]) # unique indices in pred
            gold_num = set([p[1] for p in pair[1]]) # unique indices in gold
            inter = pred_num.intersection(gold_num) # indices in common
            maxlen = max([len(pred_num),len(gold_num)])
            score = len(inter)/maxlen  # percentage of similarity
            
            if score >= self.threshold: # only add if the values are within the threshold
                pred_pol = "pos" if "Positive" in pair[0][0][0] else "neg"
                gold_pol = "pos" if "Positive" in pair[1][0][0] else "neg"
                
                pred_result.append(pred_pol) # consists of polar tags only
                gold_result.append(gold_pol) # --"--
        
        return pred_result, gold_result


    def polar_f1(self):
        """Calculates the F1-score of polar expressions alone."""
        #since we cannot compare directly, we map all chunks against the others
        #this artificially inflates the number a bit, but the ratio should not
        #make it matter that much
        localpred = []
        localgold = []

        for pred_sent, gold_sent in zip(self.pred, self.true):
            if pred_sent == gold_sent:
                continue # no need to look through
            predchunks = self.numchunks(pred_sent)
            goldchunks = self.numchunks(gold_sent)

            if predchunks == [] and goldchunks == []: # nothing to evaluate
                continue
            self.totalchunks += 1
            pred_result, gold_result = self.compare(predchunks, goldchunks)
            localpred.extend(pred_result)
            localgold.extend(gold_result)
        
        print("Number of matches",len(localpred))
        fscore_polar = f1_score(localgold, localpred, pos_label="pos")
        
        return fscore_polar


    def checkstuff(self):
        """Goes through the registered gold and pred files and outputs some statistics""" 
        
        teller = []
        for pred_sent, gold_sent in zip(self.pred,self.true):
            if pred_sent == gold_sent:
                continue #no need to look through, no expressions to evaluate
            pred_chunk = self.chunks(pred_sent)
            gold_chunk = self.chunks(gold_sent)
            teller.extend(pred_chunk)
            teller.extend(gold_chunk)
            #Only need to check consistency of pred
            self.consistency(pred_chunk)
            self.consistency(gold_chunk)
    

    def summary(self):
        """Returns number of adjacent chunks, broken chunks and mixed chunks. """

        #print("Summary of findings")
        #print("Number of cases of chunks next to each other:")
        #print("\t",self.adjacent_chunks)
        #print("Number of cases of I-tag beginning a chunk")
        #print("\t",self.broken_chunks)
        #print("Number of cases of mixed polarity in chunks")
        #print("\t",self.mixed_chunks)

        return self.adjacent_chunks, self.broken_chunks, self.mixed_chunks


    def span_fscore(self):
        """Returns strict F1-score for span, and a kind F1-score version for span.
        Strict version is no dsitinction between 'B' and 'I', while kind sets all tags to 'I'. """
        
        pred = [sent.split() for sent in self.pred]
        gold = [sent.split() for sent in self.true]

        pred_flat = [tag for sent in pred for tag in sent]
        gold_flat = [tag for sent in gold for tag in sent]

        kind_pred_normalized = [re.sub(r'([B|I]-targ-Negative|[B|I]-targ-Positive)', "I", label) for label in pred_flat]
        kind_gold_normalized = [re.sub(r'([B|I]-targ-Negative|[B|I]-targ-Positive)', "I", label) for label in gold_flat]

        # stricter version
        strict_pred_normalized = [re.sub(r'(-targ-Negative|-targ-Positive)', "", label) for label in pred_flat]
        strict_gold_normalized = [re.sub(r'(-targ-Negative|-targ-Positive)', "", label) for label in gold_flat]
        
        return f1_score(strict_gold_normalized, strict_pred_normalized, labels=["B", "I"], average="macro"), f1_score(kind_gold_normalized, kind_pred_normalized, pos_label="I") 


if __name__ == "__main__":
    #pred_baseline_path = "output/tsa_output_baseline_2024-03-08/predictions_baseline_test.txt"
    #pred_20_path = "../modeling/output/tsa_output_rulebased_20_2024-03-08/predictions_rulebased_20.txt"
    #pred_40_path = "../modeling/output/tsa_output_rulebased_40_2024-03-08/predictions_rulebased_40.txt"
    #pred_60_path = "../modeling/output/tsa_output_rulebased_60_2024-02-06/predictions_rulebased.txt"
    #pred_80_path = "../modeling/output/tsa_output_rulebased_80_2024-02-06/predictions_rulebased.txt"
    #pred_100_path = "predictions_test.txt"
    #gold_path = "true_test.txt"
    a = Analysis()
    #a.readfiles(pred_40_path, gold_path)
    #a.readfiles(pred_20_path, gold_path)
    #a.readfiles(pred_40_path, gold_path)
    #a.readfiles(pred_60_path, gold_path)
    #a.readfiles(pred_80_path, gold_path)
    #a.readfiles(pred_100_path, gold_path)
    #a.checkstuff()
    #a.summary()
    #print("Polarity f-score:", a.polar_f1())
    #print("Span f-score:", a.span_fscore())
    
