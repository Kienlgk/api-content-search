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
from DATYS.run_experiment import infer, infer_get_top_candidates, infer_get_top_candidates_v3
# importlib.reload(DATYS.run_experiment)
INDEX_380 = "test_380_with_label"


def check_datys_combine_with_ratio(a, b):
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

    for ke in ["c", "verify", "andDo", "Bibe", "Nome", "trimResults", "expireAfterWrite", "Ordering.natural", "Objects.equal"]:
        data.pop(ke)
    pop_key_list = []

    for fqn, label in data.items():
        # do datys    
        if "</API>" in fqn:
            pop_key_list.append(fqn)
    for ke in pop_key_list:
        data.pop(ke)

    with open(f"data/test_threads.json", "r") as fp:
        test_threads = json.load(fp)
    with open(f"data/test_127_threads_result.json", "r") as fp:
        classification_result = json.load(fp)

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

    with open(f"data/apis_having_emb_test_set.json", "r") as fp:
        apis = json.load(fp)

    
    count = 0
    test_apis = []
    analyse_dict = {}
    for api in apis:
        fqn = api
        if api not in data:
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

            list_mentions = infer_get_top_candidates_v3(thread_id, thread_content, thread['title'], thread['tags'], simple_name, cand_dict, scale=a)
            if len(list_mentions) == 0:
                continue
            predicted_apis = []
            predicted_scores = []
            for mention in list_mentions:
                for pred_i, pred in enumerate(mention['preds']):
                    if pred not in predicted_apis:
                        predicted_apis.append(pred)
                        predicted_scores.append(mention['score'][pred_i])
            if len(predicted_apis) == 0:
                continue

            if fqn not in predicted_apis:
                continue
            score = predicted_scores[predicted_apis.index(fqn)]
            if fqn in test_dict:
                if thread_id not in test_dict[fqn]:
                    continue
                result_ids = test_dict[fqn][thread_id]
                results = [classification_result[str(i)] for i in result_ids]
                pos_results = [res[0] for res in results]
                prob_results = [res[1][1] for res in results]
                bool_thread_refers_api = False
                sum_res = 0
                res_list = []
                for res in prob_results:
                    sum_res += res
                mean_ = sum_res/len(results)*b
                res_list.append(res)
                score = score + mean_
            if score >0.5:
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
    return Prec, Recall, F1, fps, fns, tps, count


def run_facos_ablation():
    list_prec = []
    list_recall = []
    list_f1 = []
    list_tps = []
    list_fps = []
    list_fns = []
    list_count = []
    os.makedirs("output/", exist_ok=True)
    with open(f"output/combine_scores.txt", "w+") as fp:
        for i in range(0, 11):
            a = 0.1*i
            b = 1-a
            Prec, Recall, F1, fps, fns, tps, count = check_datys_combine_with_ratio(a, b)
            list_prec.append(f"{Prec:.4f}")
            list_recall.append(f"{Recall:.4f}")
            list_f1.append(f"{F1:.4f}")
            list_fps.append(len(fps))
            list_fns.append(len(fns))
            list_tps.append(len(tps))
            list_count.append(count)
        print(f"list_prec: {list_prec}", file=fp)
        print(f"list_recall: {list_recall}", file=fp)
        print(f"list_f1: {list_f1}", file=fp)
        print(f"list_fps: {list_fps}", file=fp)
        print(f"list_fns: {list_fns}", file=fp)
        print(f"list_tps: {list_tps}", file=fp)
        print(f"list_count: {list_count}", file=fp)

if __name__ == "__main__":
    run_facos_ablation()
