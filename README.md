# FACOS: Finding API Relevant Contents on StackOverflow with Semantic and Syntactic Analysis

Collecting API examples, usages, and mentions rel- evant to a specific API method over discussions on venues such as Stack Overflow is not a trivial problem. It requires efforts to correctly recognize whether the discussion refers to the API method that developers/tools are searching for. The content of the thread, which consists of both text paragraphs describing the involvement of the API method in the discussion and the code snippets containing the API invocation, may refer to the given API method. Leveraging this observation, we develop FACOS, a context-specific algorithm to capture the semantic and syntactic information of the paragraphs and code snippets in a discussion. FACOS combines a syntactic word-based score with a score from a predictive model fine-tuned from CodeBERT. FACOS beats the approach reproduced from the state-of-the-art in API disambiguation task by 13.9% in terms of F1-score.

## 0. Download data
Create data/ folder [facos_root]/. [facos_root] is the root folder of this project.
```
$ mkdir [facos_root]/data/
$ mkdir [facos_root]/model/
```

Download data from [link](https://drive.google.com/file/d/16PPwQrpuDmEASTOSwtghNlyrL283SBtb/view?usp=sharing) and put into the data/ folder above.

Download `pytorch_model.bin` from [link](https://drive.google.com/file/d/1SrPYh3k3E9r-PKYY9rFm--QIkUgKiW_U/view?usp=sharing) and put the file into the model/ folder



## 1. Install Docker enviroment
```
$ docker compose up -d
$ docker exec -it facos bash
```

## 2. Run DATYS
```
$ cd /app/facos
$ python benchmark_datys.py
```

## 3. Run FACOS
```
$ cd /app/facos
$ python benchmark_facos.py
```

## 3.1 Regenerate the API relevance embeddings by using trained model
```
pretrained_model=microsoft/codebert-base
test_model=model/pytorch_model.bin
output_dir=facos/model/
source_length=512

python facos/rel_cls/api_rel_cls.py --model_type roberta --model_name_or_path $pretrained_model \
                                    --load_model_path $test_model --output_dir $output_dir \
                                    --max_source_length $source_length
```