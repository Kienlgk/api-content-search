# FACOS: Finding API Relevant Contents on StackOverflow with Semantic and Syntactic Analysis

Collecting API examples, usages, and mentions rel- evant to a specific API method over discussions on venues such as Stack Overflow is not a trivial problem. It requires efforts to correctly recognize whether the discussion refers to the API method that developers/tools are searching for. The content of the thread, which consists of both text paragraphs describing the involvement of the API method in the discussion and the code snippets containing the API invocation, may refer to the given API method. Leveraging this observation, we develop FACOS, a context-specific algorithm to capture the semantic and syntactic information of the paragraphs and code snippets in a discussion. FACOS combines a syntactic word-based score with a score from a predictive model fine-tuned from CodeBERT. FACOS beats the approach reproduced from the state-of-the-art in API disambiguation task by 13.9% in terms of F1-score.

## Download data
Create data/ folder in api-content-search/
```
$ mkdir api-content-search/data/
```

Download data from [link](https://drive.google.com/file/d/16PPwQrpuDmEASTOSwtghNlyrL283SBtb/view?usp=sharing) and put into the data/ folder above

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

## Updating