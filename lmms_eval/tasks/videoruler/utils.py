import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

LENGTH = [5,30,60,120,180,300,600,1200,1800,3600,7200,10800,18000]
TASK_CATEGORIES = ["QA","OCR",'temporal']







# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "videoruler.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)



def videoruler_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name,'datasets--ruili0--video_testing_dataset/snapshots/4d1971ff30f5308171ded3c04789deb7953d424d')
    # cache_dir='/pasteur/u/ruili/RULER/scripts/data/data_new_ytb_5s'
    video_path = doc["video"] 
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]




# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:




def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEF]", s):
        return ""

    matches = re.search(r"[ABCDEF]", s)
    if matches is None:
        return ""
    return matches[0]

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEF]", s):
        return ""

    matches = re.search(r"[ABCDEF]", s)
    if matches is None:
        return ""
    return matches[0]

matrices = []


for k in LENGTH:
    for l in TASK_CATEGORIES:
        matrices.append(f"{str(k)}_{l}")


def videoruler_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videoruler score), value: metric value
    """
    pred = results[0]
    task_type = doc["type"]
    if task_type=="QA" or task_type=="temporal":
        pred_ans = extract_characters_regex(pred)
    elif task_type=="OCR":
        pred_ans = re.sub(r'^\.+|\.+$', '', pred.replace("The answer is ","").split(" ")[0].split(":")[0].strip()) if pred else pred
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    
    length=doc['length']
    video=doc['video']
    data_dict = {"index": doc["index"] ,"length":length,"video":video, "task_type": task_type, "pred_answer": pred_ans, 'pred':pred,"answer": doc["answer"]}

    return {f"videoruler_perception_score": data_dict}


def videoruler_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for k in LENGTH:
        for l in TASK_CATEGORIES:
            key = f"{str(k)}_{l}"
            category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        length=result['length']
        # print(result)
        key = f"{length}_{task_type}"
        # print(key)
        category2score[key]["answered"] += 1
        # print(result["pred_answer"],result["answer"])
        category2score[key]["correct"] += result["pred_answer"].lower() == result["answer"].lower()
    print(category2score)
    for  cur_key in category2score:
        length,task_type=cur_key.split("_")
        eval_logger.info(f"Evaluation on Video Length: {str(length)} and Task: {task_type}: {100 * category2score[cur_key]['correct'] / category2score[cur_key]['answered'] if category2score[cur_key]['answered'] > 0 else 0 : .1f}%")
    for cur_length in LENGTH:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if str(cur_length) == k.split("_")[0]:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Video Length: {str(cur_length)}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
