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

from 
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
INDEX_380 = "test_380_with_label"


def read_datys_data():
    """
        This funtion helps read data provided by DATYS
    """
    fqn_threads_dict = {}
    for _f in glob.glob("/app/stackoverflow_dump/datys_data/data/so_threads/*"):
        with open(_f, "r") as fp:
            if "notjava" in _f:
                continue
            thread_id = _f.split(os.sep)[-1].split(".")[0]
            content = fp.read()
            pattern = '<API label="(.*?)">(.*?)</API>'
            for line in content.split("\n"):
                match = re.search(pattern, line)
                while match is not None:
                    s = match.start()
                    matching_tag = match.group(0)
                    label =  match.group(1).strip()
                    api = match.group(2)
                    if label == 'com.google.common.collect.Sets.difference"':
                        label = 'com.google.common.collect.Sets.difference'
                    elif label == 'com.google.common.collect.EnumBiMap.pu':
                        label = 'com.google.common.collect.EnumBiMap.put'
                    elif label == 'org.mockito.stubbing.OngoingStubbing':
                        label = 'org.mockito.stubbing.OngoingStubbing.thenReturn'
                    elif label == 'com.google.common.collect.' and thread_id == '5716267':
                        label = 'com.google.common.collect.Multimaps.index'
                    elif label == 'com.google.common.io.Closer.reclose' and thread_id == '39658005':
                        label = 'com.google.common.io.Closer.close'
                    elif label == 'om.google.common.collect.BiMap' and thread_id == '61625556':
                        label = 'com.google.common.collect.BiMap.synchronizedBiMap'
                    elif label == "org.mockito.Mockito.argThat" and thread_id == "23273230":
                        label = ""
                    elif label == "org.assertj.core.api.OptionalIntAssert." and thread_id == "48866139":
                        label = ""
                    elif label == "org.mockito.Mockito.then" and thread_id == "42082918":
                        label = "org.mockito.Mockito.when"
                    elif label == "org.mockito.stubbing.OngoingStubbing.thenThrow" and thread_id == "19155369":
                        label = "org.mockito.stubbing.OngoingStubbing.thenReturn"
                    if label != "None" and label != "":
                        if label not in fqn_threads_dict:
                            fqn_threads_dict[label] = []
                        if thread_id not in fqn_threads_dict[label]:
                            fqn_threads_dict[label].append(thread_id)
                        
                    line = re.sub(re.escape(matching_tag), api, line, 1)
                    match = re.search(pattern, line)
    return fqn_threads_dict

def benchmark_datys():
    is_train_set = False
    IS_PRINT = False
    INDEX_380 = "test_380_with_label"
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

    with open('../api_method_candidates.json', "r") as fp:
        api_method_candidates = json.load(fp)

    for k in ["c", "verify", "andDo", "Bibe", "Nome", "trimResults", "expireAfterWrite", "Ordering.natural", "Objects.equal"]:
        data.pop(k)
    pop_key_list = []

    for fqn, label in data.items():
        # do datys    
        if "</API>" in fqn:
            pop_key_list.append(fqn)
    for k in pop_key_list:
        data.pop(k)


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
        class_name= fqn.split(".")[-2]

        threads, total_matches = es_client.query(INDEX_380, f"{simple_name} {simple_name}(", op="or", field="text")
        other_threads, total_matches = es_client.query(INDEX_380, f"{simple_name} {simple_name}(", op="or", field="code")
        threads = [thread['_id'] for thread in threads]
        other_threads = [thread['_id'] for thread in other_threads if thread['_id'] not in threads]
        threads = threads + other_threads
        threads = [thrd for thrd in threads if thrd in test_threads]

        
        if fqn not in predictions:
            predictions[fqn] = []

        for thread_id in threads:
            thread = es_client.get(thread_id, INDEX_380)
            cand_dict = {}

            cand_dict = api_method_candidates[simple_name]
            thread_path = glob.glob("/app/stackoverflow_dump/datys_data/data/so_threads/"+str(thread_id)+".*")[0]
            with open(thread_path, "r") as tfp:
                thread_content = tfp.read()
                
            list_mentions = infer_get_top_candidates(thread_id, thread_content, thread['title'], thread['tags'], simple_name, cand_dict)
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