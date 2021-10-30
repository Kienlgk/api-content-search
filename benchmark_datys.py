import os
from pprint import pprint
import time
import traceback
import csv
import numpy as np
import glob
import json
import re
from html.parser import HTMLParser
from html.entities import entitydefs
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError as XmlParseError
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from utils.benchmark_result import BaselineResult
from utils.thread_reader import LabeledThreadReader

from utils.data_utils import read_datys_data
import sys, os.path
sys.path.append(os.path.abspath('/app/src/utils/'))
sys.path.append(os.path.abspath('/app/src/'))

INDEX_NAME = "so_java_threads"
host = "elasticsearch"
from es_client import ESClient
es_client = ESClient(index=INDEX_NAME, scroll_loop_limit=10, host=host, port=9200)

import DATYS
from DATYS.run_experiment import infer, infer_get_top_candidates
# importlib.reload(DATYS.run_experiment)


def benchmark_datys():
    is_train_set = False
    IS_PRINT = False
    do_print_failed_cases = True
    data = read_datys_data()
    predictions  = {}
    predictions_tp = {}
    all_baseline_result = []
    break_count = 0
    IS_PRINT = False
    fps = []
    fns = []
    tps = []
    failed_cases = []

    with open('data/api_method_candidates.json', "r") as fp:
        api_method_candidates = json.load(fp)
    with open(f"data/test_threads.json", "r") as fp:
        test_threads = json.load(fp)
    with open(f"data/apis_having_emb_test_set.json", "r") as fp:
        apis = json.load(fp)

    test_dict = {}
    with open(f"data/text_code_pairs_test.jsonl", "r") as fp:
        for line in fp.readlines():
            stack_pairs = json.loads(line)
            _idx = stack_pairs['idx']
            thread_id = stack_pairs['thread_id']
            simple_name = stack_pairs['simple_name']
            target_fqn = stack_pairs['target_fqn']
            cls_label = stack_pairs['cls_label']
            if target_fqn not in test_dict:
                test_dict[target_fqn] = {}
            if thread_id not in test_dict[target_fqn]:
                test_dict[target_fqn][thread_id]= []
            if _idx not in test_dict[target_fqn][thread_id]:
                test_dict[target_fqn][thread_id].append(_idx)

    
    with open("data/search_threads_from_es.json", "r") as fp:
        queried_threads = json.load(fp)
    with open("data/title.json", "r") as fp:
        thread_titles = json.load(fp)
    with open("data/tags.json", "r") as fp:
        thread_tags = json.load(fp)

    count = 0
    test_apis = []
    analyse_dict = {}
    for api in apis:
        fqn = api
        if api not in data:
            continue
        if api not in test_dict:
            continue
        labels_ = data[api]
        labels_ = [lbl for lbl in labels_ if lbl in test_threads]

        simple_name = fqn.split(".")[-1]

        threads = queried_threads[api]
        
        if fqn not in predictions:
            predictions[fqn] = []

        for thread_id in threads:
            thread_title = thread_titles[thread_id]
            thread_tag = thread_tags[thread_id]
            cand_dict = {}

            cand_dict = api_method_candidates[simple_name]
            thread_path = glob.glob("data/so_threads/"+str(thread_id)+".*")[0]

            with open(thread_path, "r") as tfp:
                thread_content = tfp.read()
                
            list_mentions = infer_get_top_candidates(thread_id, thread_content, thread_title, thread_tag, simple_name, cand_dict)
            predicted_apis = list_mentions[0]['preds']
            if fqn in predicted_apis:
                predictions[fqn].append(thread_id)
        # """
        if IS_PRINT: print("FQN: ", fqn)
        if IS_PRINT: print("Top-5 threads: ", predictions[fqn][:5])
        if IS_PRINT: print("labels: ", labels_)
        benchmark_result = BaselineResult(fqn, labels_, predictions[fqn])
        fp_cases = benchmark_result.get_fp_cases()
        fn_cases = benchmark_result.get_fn_cases()
        tp_cases = benchmark_result.get_tp_cases()

        fps += fp_cases
        fns += fn_cases
        tps += tp_cases
        if fqn not in analyse_dict:
            analyse_dict[fqn] = {'fn': [], 'fp': []}
        analyse_dict[fqn]['fn'] += fn_cases
        analyse_dict[fqn]['fp'] += fp_cases
        count += 1
        test_apis.append(api)
        predictions_tp[fqn] = benchmark_result.get_tp_cases()
        all_baseline_result.append(benchmark_result)
        
        do_print_failed_cases = False
        if do_print_failed_cases: print("fp: ", fp_cases)
        if do_print_failed_cases: print("fn: ", fn_cases)
        if do_print_failed_cases: print("tp: ", tp_cases)
        if do_print_failed_cases: print("prec:", benchmark_result.prec)
        if do_print_failed_cases: print("recall:", benchmark_result.recall)
        if do_print_failed_cases: print("f1:", benchmark_result.f1)

    Prec = sum([result.prec for result in all_baseline_result])/len(all_baseline_result)
    Recall = sum([result.recall for result in all_baseline_result])/len(all_baseline_result)
    F1 = sum([result.f1 for result in all_baseline_result])/len(all_baseline_result)


    print("Nrof results: ",len(all_baseline_result))
    print("avg Prec: ", Prec)
    print("avg Recall: ", Recall)
    print("avg F1: ", F1)
    print("fps:", len(fps))
    print("fns:", len(fns))
    print("tps:", len(tps))
    print(f"Nrof APIs: {count}")


if __name__ == "__main__":
    benchmark_datys()