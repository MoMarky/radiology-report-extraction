import os
import os.path as osp
import argparse
import csv
import numpy
import matplotlib
from matplotlib import pyplot, image
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from shutil import copyfile
import re
import xml.dom.minidom
import pandas as pd
import time
import re
import random
import json
import re
import matplotlib.pyplot as pl
from PIL import Image
from collections import defaultdict
import pickle
import gzip
import random
import sys
from pathlib import Path

def extract_findings_and_impression_openi():
    root = r"G:\Medical Reports datasets\NLMCXR_reports\ecgen-radiology"

    valid_count = 0

    list_writer = csv.writer(
        open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_report_sections_all.csv", 'a+', encoding='utf-8', newline=""))
    list_writer.writerow(["ids", "finding", "impression"])

    for file_name in tqdm(os.listdir(root)):
        dom = xml.dom.minidom.parse(osp.join(root, file_name))
        file_id = file_name.split('.')[0]
        try:
            cc = dom.getElementsByTagName('AbstractText')
            c1 = cc[2]
            finding = c1.firstChild.data
            c1 = cc[3]
            impression = c1.firstChild.data
            list_writer.writerow([file_id, finding, impression])
            valid_count += 1
        except AttributeError:
            continue

def gen_openi_data():
    all_label_csv = open(osp.join(r"G:\Medical Reports datasets\NLMCXR_reports\openi_findings_label.csv"), "r").readlines()

    all_ids = []
    for line in all_label_csv[1:]:
        id = line.split(',')[0]

        all_ids.append(int(id))

    random.shuffle(all_ids)
    random.shuffle(all_ids)
    # print(all_ids)

    test_576_ids = random.sample(all_ids, 576)

    train_ids = []

    for ii in all_ids:
        if ii not in test_576_ids:
            train_ids.append(ii)

    save_dict = {"test": test_576_ids, "train": train_ids}
    print('test', len(test_576_ids), 'train', len(train_ids))
    json.dump(save_dict, open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_random_split.json", 'w'))

def gen_report_and_label():
    save_root = r"G:\Medical Reports datasets\NLMCXR_reports\test_576"

    data_csv = pd.read_csv(r"G:\Medical Reports datasets\NLMCXR_reports\openi_report_sections_all.csv")
    label_csv = pd.read_csv(r"G:\Medical Reports datasets\NLMCXR_reports\openi_findings_label.csv")
    fold_json = json.load(open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_random_split.json", 'r'))

    header_all = open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_findings_label.csv", "r").readlines()[0].split(',')
    header_label = [header_all[0]] + header_all[2:]
    header_data = open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_report_sections_all.csv", "r").readlines()[
        0].split(',')

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    test_ids = fold_json["test"]

    test_data_df = pd.DataFrame(data=None, columns=header_data)
    test_label_df = pd.DataFrame(data=None, columns=header_label)

    train_data_df = pd.DataFrame(data=None, columns=header_data)
    train_label_df = pd.DataFrame(data=None, columns=header_label)

    for index, row in data_csv.iterrows():
        report_id = int(row["ids"])

        if report_id in test_ids:
            test_findings = row["finding"]
            test_impression = row["impression"]
            data_line_df = pd.DataFrame([[report_id, test_findings, test_impression]], columns=header_data)
            test_data_df = test_data_df.append(data_line_df, ignore_index=True)

            test_label_line = label_csv.iloc[index]
            label = np.array(test_label_line[2:])
            label = label.astype('float')
            label[np.isnan(label)] = 2

            label_line_df = pd.DataFrame([[report_id] + list(label)], columns=header_label)
            test_label_df = test_label_df.append(label_line_df, ignore_index=True)
        else:
            train_findings = row["finding"]
            train_impression = row["impression"]
            data_line_df = pd.DataFrame([[report_id, train_findings, train_impression]], columns=header_data)
            train_data_df = train_data_df.append(data_line_df, ignore_index=True)

            train_label_line = label_csv.iloc[index]
            label = np.array(train_label_line[2:])
            label = label.astype('float')
            label[np.isnan(label)] = 2

            label_line_df = pd.DataFrame([[report_id] + list(label)], columns=header_label)
            train_label_df = train_label_df.append(label_line_df, ignore_index=True)

    test_data_df.to_csv(osp.join(save_root, "test_data.csv"), index=False)
    test_label_df.to_csv(osp.join(save_root, "test_label.csv"), index=False)

    train_data_df.to_csv(osp.join(save_root, "train_data.csv"), index=False)
    train_label_df.to_csv(osp.join(save_root, "train_label.csv"), index=False)



if __name__ == '__main__':
    # extract_findings_and_impression_openi()
    # gen_openi_data()
    gen_report_and_label()

