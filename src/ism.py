"""Image sequence manipulator, a tool for working with the UI training data.

UI training data is stored in a folder as a numbered image sequence, along with an info.json and
a labels.csv. info.json stores information about the images - currently, just their source video.
labels.csv maps every image to an image class used to train a neural network.
"""

from typing import Dict, Any, List
import shutil
import os
import json
import csv
import math
import cv2
import numpy as np

def load_size(ui_folder: str) -> Dict[str, int]:
    with open(ui_folder + "/size.json") as size_file:
        return json.load(size_file)

def load_info(folder: str) -> Dict[str, Any]:
    info_file = open(folder + "/info.json")
    info = json.load(info_file)
    info_file.close()
    return info

def write_info(folder: str, info: Dict[str, Any]) -> None:
    info_file = open(folder + "/info.json", "w")
    json.dump(info, info_file, indent=4)
    info_file.close()

def load_labels(folder: str) -> List[List[str]]:
    label_file = open(folder + "/labels.csv")
    labels = []
    for row in csv.reader(label_file):
        labels.append(row)
    
    label_file.close()
    return labels

def write_labels(folder: str, labels: List[List[str]]) -> None:
    label_file = open(folder + "/labels.csv", "w", newline="")
    csv.writer(label_file).writerows(labels)
    label_file.close()

def defragment(folder: str) -> None:
    labels = load_labels(folder)
    
    num_digits = len(labels[0][0])
    for i in range(0, len(labels)):
        image_path = f"{folder}/{labels[i][0]}.png"
        if int(labels[i][0]) > i:
            new_name = str(i).zfill(num_digits)
            os.rename(image_path, f"{folder}/{new_name}.png")
            labels[i][0] = new_name
    
    write_labels(folder, labels)

def set_num_digits(folder: str, new_num_digits: int = 0) -> None:
    labels = load_labels(folder)
    num_digits = len(labels[0][0])
    digits_lower_bound = math.ceil(math.log10(int(labels[-1][0]) + 1))
    
    if new_num_digits < digits_lower_bound:
        new_num_digits = digits_lower_bound
    
    if not new_num_digits == num_digits:
        for label in labels:
            image_path = f"{folder}/{label[0]}.png"
            label[0] = label[0].lstrip('0').zfill(new_num_digits)
            os.rename(image_path, f"{folder}/{label[0]}.png")
    
    write_labels(folder, labels)

def shift(folder: str, amount: int) -> None:
    if not amount == 0:
        labels = load_labels(folder)
        if int(labels[0][0]) + amount < 0:
            raise ValueError("Can't shift image numbers that low")
        
        max_image_num = int(labels[-1][0]) + amount
        new_num_digits = math.ceil(math.log10(max_image_num + 1))
        
        r = range(0, len(labels))
        if amount > 0:
            r = reversed(r)
        
        for i in r:
            image_path = f"{folder}/{labels[i][0]}.png"
            labels[i][0] = labels[i][0].lstrip('0').zfill(new_num_digits)
            os.rename(image_path, f"{folder}/{labels[i][0]}.png")
        
        write_labels(folder, labels)

def delete_similar(folder: str, window_size: int = 1, threshold: float = 0.0) -> None:
    labels = load_labels(folder)
    num_digits = len(labels[0][0])
    
    total_comparisons = (len(labels) - window_size) * window_size + window_size * (window_size - 1) // 2
    comparisons = 0
    
    for i in reversed(range(1, len(labels))):
        print(f"{100 * comparisons // total_comparisons}%  {comparisons} / {total_comparisons}")
        comparisons += min(window_size, i)
        
        image1_path = f"{folder}/{labels[i][0]}.png"
        image1 = cv2.imread(image1_path)
        width = image1.shape[0]
        height = image1.shape[1]
        
        for j in reversed(range(max(i - window_size, 0), i)):
            image2 = cv2.imread(f"{folder}/{labels[j][0]}.png")
            diff = cv2.absdiff(image1, image2)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            num_diff_pxls = cv2.countNonZero(diff)
            diff_score = 0.0 if num_diff_pxls == 0 else cv2.mean(diff)[0] * width * height / num_diff_pxls
            if diff_score < threshold:
                os.remove(image1_path)
                labels.pop(i)
                break
    
    write_labels(folder, labels)

def resize(ui_folder: str, scene: str, in_scale: int, out_scale: int) -> None:
    size = load_size(ui_folder)
    width = size["width"] // out_scale
    height = size["height"] // out_scale
    
    in_folder = f"{ui_folder}/training_data/scale{in_scale}/{scene}"
    out_folder = f"{ui_folder}/training_data/scale{out_scale}/{scene}"
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    shutil.copyfile(in_folder + "/info.json", out_folder + "/info.json")
    
    labels = load_labels(in_folder)
    count = len(labels)
    write_labels(out_folder, labels)
    
    for i in range(0, count):
        print(f"{100 * i // count}%  {i} / {count}")
        image = cv2.imread(f"{in_folder}/{labels[i][0]}.png")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{out_folder}/{labels[i][0]}.png", image)
