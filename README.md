# DA-for-Norwegian-TSA
This is the repository related to the thesis titled "Data Augentation for Norwegian Targeted Sentiment Analysis". Please refer to the thesis (link is coming) for more information, as well as citation. In this repository, we provide the code behind the rule-based and model-based methods, as well as four new augmented training datasets, derived from the NoReC_TSA dataset (https://github.com/ltgoslo/norec_tsa).

## Folder Structure
* **data** contains all the augmented data created with rulebased and model-based methods.
* **methods** contains the code used to produce the data, the performance results and perform error analysis. This folders is divided into three folders: **baseline** contains the files used to run the NB-BERT model, which is used for both the baseline, rule-based and generative model-based methods. The **generative** folder contains the code used to produce the augmented data through generative prompts with ChatNorT5, and the **rulebased** folder contain the files used to produce augmented data using a rule-based method. NorSentLex (https://github.com/ltgoslo/norsentlex) and Norsk Ordbank (https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-5/) are essential resources in this folder. 

Note that this repository is not optimized for code reproducibility, and that there are paths that need to be changed if one wishes to do so, as well as downloading the necessary models.  
 
