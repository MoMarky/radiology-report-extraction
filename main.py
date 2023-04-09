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

"""MIMIC"""
def find_all_test():
    split_csv = r"C:\\Users\\M S I\\Desktop\\mimic-cxr-2.0.0-split.csv"
    split_lines = open(split_csv, 'r').readlines()

    list_writer = csv.writer(open(r"G:\Medical Reports datasets\mimic_report_val\all_test_ids.csv", 'a+', encoding='utf-8', newline=""))

    test_sets = []
    for line in tqdm(split_lines[1:]):
        if 'test' in line:
            write_line = line.rstrip().split(',')
            title_line = []
            for item in write_line:
                title_line.append(item)
            list_writer.writerow(title_line)
def find_all_train():
    split_csv = r"C:\\Users\\M S I\\Desktop\\mimic-cxr-2.0.0-split.csv"
    split_lines = open(split_csv, 'r').readlines()

    list_writer = csv.writer(open(r"G:\Medical Reports datasets\mimic_report_val\all_train_ids.csv", 'a+', encoding='utf-8', newline=""))

    test_sets = []
    for line in tqdm(split_lines[1:]):
        if 'train' in line:
            write_line = line.rstrip().split(',')
            title_line = []
            for item in write_line[1:]:
                title_line.append(item)
            list_writer.writerow(title_line)


def test1():
    split_csv = r"C:\\Users\\M S I\\Desktop\\mimic-cxr-2.0.0-split.csv"
    split_lines = open(split_csv, 'r').readlines()

    val_sets = []
    for line in tqdm(split_lines[1:]):
        if 'validate' in line:


            val_sets.append(line)

    print(len(val_sets))

    reports_root = r"G:\\Medical Reports datasets\\mimic-cxr-reports\\files\\"
    save_report_root = r"G:\\Medical Reports datasets\\mimic_report_val\\reports\\"
    for val_line in tqdm(val_sets):
        study_id, subject_id = val_line.split(',')[1], val_line.split(',')[2]

        root_p = 'p{}'.format(subject_id[:2])
        full_p = 'p{}'.format(subject_id)
        report_name = 's{}.txt'.format(study_id)



        report_path = osp.join(reports_root, root_p, full_p, report_name)

        if not os.path.isfile(report_path):
            print(report_path)
            continue

        new_report_name = "p{}_s{}.txt".format(subject_id, study_id)

        copyfile(report_path, osp.join(save_report_root, new_report_name))


def gen_label():
    root = r"G:\Medical Reports datasets\mimic_report_val\reports"

    # chexpert_csv = r"C:\\Users\\M S I\\Desktop\\mimic-cxr-2.0.0-chexpert.csv"
    chexpert_csv = r"C:\\Users\\M S I\\Desktop\\mimic-cxr-2.0.0-negbio.csv"
    chexpert_lines = open(chexpert_csv, 'r').readlines()

    list_writer = csv.writer(open(r"G:\Medical Reports datasets\mimic_report_val\negbio_labels.csv", 'a+', encoding='utf-8', newline=""))

    title_line = []
    for item in chexpert_lines[0].rstrip().split(','):
        title_line.append(item)
    list_writer.writerow(title_line)
    for txt_name in tqdm(os.listdir(root)):
        pid, sid = txt_name.split('_')[0][1:], txt_name.split('_')[1].split('.')[0][1:]

        for line in chexpert_lines[1:]:
            if pid in line and sid in line:
                # print(line)

                write_line = line.rstrip().split(',')
                title_line = []
                for item in write_line:
                    title_line.append(item)
                list_writer.writerow(title_line)
    list_writer = None


def count_key_words():
    root = r"G:\Medical Reports datasets\mimic_report_val\keywords_v1\re\content\drive\MyDrive\ChestGPT\MIMIC_Reports\reports_keywords_100"

    txt_files = [file for file in os.listdir(root) if file.endswith(".txt")]

    # class_names = ['Enlarged Cardiomegaly', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
    # Consolidation: None
    # Pneumonia: None
    # Atelectasis: None
    # Pneumothorax: None
    # Pleural Effusion: None
    # Pleural Other: None
    # Fracture: None
    # Support Devices: None']
    #
    #     for file_name in tqdm(txt_files):
    #         with open(os.path.join(root, file_name), "r") as f:
    #             content = f.read()

def extract_findings_and_impression_v2():
    test_ids = open(r"G:\Medical Reports datasets\mimic_report_val\all_train_ids.csv", 'r').readlines()
    # test_ids = open(r"G:\Medical Reports datasets\mimic_report_val\all_test_ids_2.csv", 'r').readlines()

    root = r"G:\Medical Reports datasets\mimic-cxr-reports\files"
    valid_count = 0

    list_writer = csv.writer(
        open(r"G:\Medical Reports datasets\mimic_report_val\mimic_report_sections_train_v4.csv", 'a+', encoding='utf-8', newline=""))
        # open(r"G:\Medical Reports datasets\mimic_report_val\mimic_report_sections_test_v6.csv", 'a+', encoding='utf-8', newline=""))
    # list_writer.writerow(["img_name", "subject_id", "study_id", "finding", "impression"])
    list_writer.writerow(["subject_id", "study_id", "finding", "impression"])
    # for p_id in os.listdir(root):
    #     p_subject_root = osp.join(root, p_id)
    #
    #     for p_subject in os.listdir(p_subject_root):
    #         s_study_root = osp.join(p_subject_root, p_subject)
    #
    #         for study_names in os.listdir(s_study_root):
    #             print(os.path.join(s_study_root, study_names))

    for test_line in tqdm(test_ids):
        # img_name = test_line.split(',')[0]
        subject_id = test_line.split(',')[1]
        study_id = test_line.split(',')[0]

        txt_path = osp.join(root, "p{}".format(subject_id[:2]), "p{}".format(subject_id), "s{}.txt".format(study_id))
        # print(txt_path)
        with open(txt_path, "r") as f:
                content = f.read()

        if 'FINDINGS:' in content and 'IMPRESSION:' in content:
            valid_count += 1

            finding_start_index = content.find('FINDINGS:')
            imp_start_index = content.find('IMPRESSION:')
            if finding_start_index != -1 and imp_start_index != -1:

                if content.count("FINDINGS:") > 1:
                    finding_start_index = content.rfind("FINDINGS:")
                    content = content[finding_start_index:]
                    imp_start_index = content.find('IMPRESSION:')
                    if imp_start_index == -1:
                        print(txt_path, "cant find impression after findings", 'finding_start_index', finding_start_index, 'imp_start_index', imp_start_index)
                        # exit(0)
                        continue

                    finding_text = content[9: imp_start_index]
                    imp_text = extract_imp_base_start(content, imp_start_index)

                elif content.count("IMPRESSION:") > 1 and content.count("FINDINGS:") == 1:
                    imp_start_index = content.rfind("IMPRESSION:")
                    finding_text = content[finding_start_index + 9: imp_start_index]
                    imp_text = extract_imp_base_start(content, imp_start_index)
                else:
                    finding_text = content[finding_start_index + 9: imp_start_index]
                    imp_text = extract_imp_base_start(content, imp_start_index)

                imp_text = imp_text.replace("\n", "").lstrip().rstrip()
                finding_text = finding_text.replace("\n", "").lstrip().rstrip()

                if len(imp_text.split(" ")) <= 2:
                    print(txt_path, 'too small imp_text', imp_text)
                    continue
                if len(finding_text.split(" ")) <= 10:
                    print(txt_path, 'too small finding_text', finding_text)
                    continue

                list_writer.writerow([subject_id, study_id, finding_text, imp_text])
                # list_writer.writerow([img_name, subject_id, study_id, finding_text, imp_text])
            else:
                print(txt_path, 'finding_start_index', finding_start_index, 'imp_start_index', imp_start_index)

        else:
            continue
    print(valid_count)

def extract_imp_base_start(content, start_index):
    condidate_imp_text = content[start_index + 11:]

    if ": " in condidate_imp_text:
        impression_area_end_index = condidate_imp_text.find(":")
        impression_area = condidate_imp_text[:impression_area_end_index]

        end_index = impression_area.rfind("\n")
        imp_text = impression_area[:end_index]
        imp_text = imp_text.replace("\n", "").lstrip().rstrip()
        # print(imp_text)
    else:
        imp_text = condidate_imp_text
        imp_text = imp_text.replace("\n", "").lstrip().rstrip()
        # print(imp_text)

    """delete note like 'talk to Doc, at pm, where'"""
    sentence_list = imp_text.split('.')
    # "with Dr", "by Dr", "to Dr",
    key_words = ["email", "phone", "Dr", "contact", "discuss", "minutes", "review", "dictation", "observation", "communi"]

    first_cut_pos = 0
    temp_key = []
    for sentence_index, single_sentence in enumerate(sentence_list):
        for keyy in key_words:
            if keyy in single_sentence:
                temp_key.append(keyy)
                find_pos = single_sentence.find(keyy)
                # first_cut_pos += find_pos
                break
            # else:
            #     if len(temp_key) != 0:
            #         first_cut_pos += len(single_sentence)
        if len(temp_key) != 0:
            break

    if sentence_index == 0:
        if len(temp_key) == 0:
            first_cut_pos = len(imp_text)
        else:
            first_cut_pos = find_pos
    else:
        for ii in range(sentence_index):
            first_cut_pos += (len(sentence_list[ii]) + 1)

    new_imp_text = imp_text[:first_cut_pos]

    return new_imp_text


def gen_14_label_based_on_csv_train():
    # data_csv = pd.read_csv(r"G:\Medical Reports datasets\mimic_report_val\mimic_report_sections_train_v2.csv")
    # label_csv = pd.read_csv(r"C:\Users\M S I\Desktop\mimic-cxr-2.0.0-chexpert.csv")
    # list_writer = csv.writer(open(r"G:\Medical Reports datasets\mimic_report_val\train_v2_labels.csv", 'a+', encoding='utf-8', newline=""))
    # list_writer.writerow(open(r"C:\Users\M S I\Desktop\mimic-cxr-2.0.0-chexpert.csv", "r").readlines()[0].split(','))

    data_csv = pd.read_csv(r"G:\Medical Reports datasets\mimic_report_val\mimic_report_sections_train_v4.csv")
    label_csv = pd.read_csv(r"E:\0_my_exp\ChestGPT\mimic-cxr-2.0.0-chexpert.csv")
    # list_writer = csv.writer(open(r"G:\Medical Reports datasets\mimic_report_val\test_v4_labels.csv", 'a+', encoding='utf-8', newline=""))
    header = open(r"E:\0_my_exp\ChestGPT\mimic-cxr-2.0.0-chexpert.csv", "r").readlines()[0].split(',')

    df = pd.DataFrame(data=None, columns=header)

    for index, row in tqdm(data_csv.iterrows()):
        data_study_id = row["study_id"]

        for ii, ii_row in label_csv.iterrows():
            if data_study_id == ii_row["study_id"]:
                label = np.array(ii_row[2:])
                label[np.isnan(label)] = 2

                line_df = pd.DataFrame([[ii_row["subject_id"], ii_row["study_id"]] + list(label)], columns=header)
                df = df.append(line_df, ignore_index=True)
                # list_writer.writerow([ii_row["subject_id"], ii_row["study_id"]] + list(label))
                break
    # del list_writer
    df.to_csv(r"G:\Medical Reports datasets\mimic_report_val\train_v4_labels.csv")



"""OPEN I"""
def extract_findings_and_impression_openi():
    root = r"G:\Medical Reports datasets\NLMCXR_reports\ecgen-radiology"

    valid_count = 0

    list_writer = csv.writer(
        open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_report_sections_all.csv", 'a+', encoding='utf-8', newline=""))
    # list_writer.writerow(["img_name", "subject_id", "study_id", "finding", "impression"])
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


def gen_openi_10fold_data():
    all_label_csv = open(osp.join(r"G:\Medical Reports datasets\NLMCXR_reports\openi_findings_label.csv"), "r").readlines()

    all_ids = []
    for line in all_label_csv[1:]:
        id = line.split(',')[0]

        all_ids.append(int(id))

    random.shuffle(all_ids)
    random.shuffle(all_ids)
    # print(all_ids)

    all_len = len(all_ids)
    inter = all_len // 10

    save_dict = {}
    for i in range(10):
        test_list = all_ids[i*inter:(i+1)*inter]
        save_dict.update({i: test_list})

    json.dump(save_dict, open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_10_flod_ids.json", 'w'))

def split_10_fold_report_and_label():
    root = r"G:\Medical Reports datasets\NLMCXR_reports\10_fold"

    data_csv = pd.read_csv(r"G:\Medical Reports datasets\NLMCXR_reports\openi_report_sections_all.csv")
    label_csv = pd.read_csv(r"G:\Medical Reports datasets\NLMCXR_reports\openi_findings_label.csv")
    fold_json = json.load(open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_10_flod_ids.json", 'r'))

    header_all = open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_findings_label.csv", "r").readlines()[0].split(',')
    header_label = [header_all[0]] + header_all[2:]
    header_data = open(r"G:\Medical Reports datasets\NLMCXR_reports\openi_report_sections_all.csv", "r").readlines()[0].split(',')
    for fold_i in tqdm(range(10)):
        save_root = osp.join(root, str(fold_i))
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        test_ids = fold_json[str(fold_i)]

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

        test_data_df.to_csv(osp.join(save_root, "fold{}_test_data.csv".format(fold_i)), index=False)
        test_label_df.to_csv(osp.join(save_root, "fold{}_test_label.csv".format(fold_i)), index=False)

        train_data_df.to_csv(osp.join(save_root, "fold{}_train_data.csv".format(fold_i)), index=False)
        train_label_df.to_csv(osp.join(save_root, "fold{}_train_label.csv".format(fold_i)), index=False)



if __name__ == '__main__':
    # test()
    # extract_findings_and_impression()
    # find_all_test()
    # test_ids = open(r"G:\Medical Reports datasets\mimic_report_val\mimic_report_sections_train.csv", 'r').readlines()
    # find_all_train()
    # extract_findings_and_impression_v2()
    # debug_extrct_method()

    # check()
    # extract_findings_and_impression_openi()

    # gen_14_label_based_on_csv()
    # gen_14_label_based_on_csv_train()

    """OpenI"""

    # gen_openi_10fold_data()
    split_10_fold_report_and_label()