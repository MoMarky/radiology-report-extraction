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


# local folder import
import section_parser as sp


no_split = True
reports_root_path = r"mimic-cxr-reports\files"
output_path = r""

def list_rindex(l, s):
    """Helper function: *last* matching element in a list"""
    return len(l) - l[-1::-1].index(s) - 1


def extract_sections():
    """This code is implemented by MIT-LCP at https://github.com/MIT-LCP/mimic-cxr"""

    if not osp.exists(output_path):
        os.makedirs(output_path)

    # not all reports can be automatically sectioned
    # we load in some dictionaries which have manually determined sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    # get all higher up folders (p00, p01, etc)
    p_grp_folders = os.listdir(reports_root_path)
    p_grp_folders = [p for p in p_grp_folders
                     if p.startswith('p') and len(p) == 3]
    p_grp_folders.sort()

    # patient_studies will hold the text for use in NLP labeling
    patient_studies = []

    # study_sections will have an element for each study
    # this element will be a list, each element having text for a specific section
    study_sections = []
    for p_grp in p_grp_folders:
        # get patient folders, usually around ~6k per group folder
        cxr_path = osp.join(reports_root_path, p_grp)
        p_folders = os.listdir(cxr_path)
        p_folders = [p for p in p_folders if p.startswith('p')]
        p_folders.sort()

        # For each patient in this grouping folder
        print(p_grp)
        for p in tqdm(p_folders):
            patient_path = osp.join(cxr_path, p)

            # get the filename for all their free-text reports
            studies = os.listdir(patient_path)
            studies = [s for s in studies
                       if s.endswith('.txt') and s.startswith('s')]

            for s in studies:
                # load in the free-text report
                with open(osp.join(patient_path, s), 'r') as fp:
                    text = ''.join(fp.readlines())

                # get study string name without the txt extension
                s_stem = s[0:-4]

                # custom rules for some poorly formatted reports
                if s_stem in custom_indices:
                    idx = custom_indices[s_stem]
                    patient_studies.append([s_stem, text[idx[0]:idx[1]]])
                    continue

                # split text into sections
                sections, section_names, section_idx = sp.section_text(
                    text
                )

                # check to see if this has mis-named sections
                # e.g. sometimes the impression is in the comparison section
                if s_stem in custom_section_names:
                    sn = custom_section_names[s_stem]
                    idx = list_rindex(section_names, sn)
                    patient_studies.append([s_stem, sections[idx].strip()])
                    continue

                # grab the *last* section with the given title
                # prioritizes impression > findings, etc.

                # "last_paragraph" is text up to the end of the report
                # many reports are simple, and have a single section
                # header followed by a few paragraphs
                # these paragraphs are grouped into section "last_paragraph"

                # note also comparison seems unusual but if no other sections
                # exist the radiologist has usually written the report
                # in the comparison section
                idx = -1
                for sn in ('impression', 'findings',
                           'last_paragraph', 'comparison'):
                    if sn in section_names:
                        idx = list_rindex(section_names, sn)
                        break

                if idx == -1:
                    # we didn't find any sections we can use :(
                    patient_studies.append([s_stem, ''])
                    print(f'no impression/findings: {patient_path / s}')
                else:
                    # store the text of the conclusion section
                    patient_studies.append([s_stem, sections[idx].strip()])

                study_sectioned = [s_stem]
                for sn in ('impression', 'findings',
                           'last_paragraph', 'comparison'):
                    if sn in section_names:
                        idx = list_rindex(section_names, sn)
                        study_sectioned.append(sections[idx].strip())
                    else:
                        study_sectioned.append(None)
                study_sections.append(study_sectioned)
    # write distinct files to facilitate modular processing
    if len(patient_studies) > 0:
        # write out a single CSV with the sections
        with open(osp.join(output_path, 'mimic_cxr_sectioned.csv'), 'w') as fp:
            csvwriter = csv.writer(fp)
            # write header
            csvwriter.writerow(['study', 'impression', 'findings',
                                'last_paragraph', 'comparison'])
            for row in study_sections:
                csvwriter.writerow(row)

        if no_split:
            # write all the reports out to a single file
            with open(osp.join(output_path, f'mimic_cxr_sections.csv'), 'w') as fp:
                csvwriter = csv.writer(fp)
                for row in patient_studies:
                    csvwriter.writerow(row)
        else:
            # write ~22 files with ~10k reports each
            n = 0
            jmp = 10000

            while n < len(patient_studies):
                n_fn = n // jmp
                with open(osp.join(output_path, f'mimic_cxr_{n_fn:02d}.csv'), 'w') as fp:
                    csvwriter = csv.writer(fp)
                    for row in patient_studies[n:n+jmp]:
                        csvwriter.writerow(row)
                n += jmp


def select_test_data_from_sections():
    all_sections = pd.read_csv(r"\mimic_cxr_sectioned.csv")

    test_ids = open(r"all_test_ids.csv", 'r').readlines()

    list_writer = csv.writer(
        open(r"mimic_baseline_sections_test.csv", 'a+', encoding='utf-8', newline=""))
    list_writer.writerow(["subject_id", "study_id", "finding", "impression"])

    for test_line in tqdm(test_ids):
        # img_name = test_line.split(',')[0]
        test_subject_id = test_line.split(',')[1]
        test_study_id = test_line.split(',')[0]

        for index, row in all_sections.iterrows():
            study_id = row["study"][1:]

            if study_id == test_study_id:
                list_writer.writerow([test_subject_id, study_id, row["findings"], row["impression"]])
                break

def data_clean_for_mimic():
    new_test_data = pd.read_csv(r"G:\Medical Reports datasets\mimic_report_val\new_baseline_data\mimic_baseline_sections_test.csv")

    list_writer = csv.writer(
        open(r"G:\Medical Reports datasets\mimic_report_val\new_baseline_data\mimic_baseline_sections_test_V2_Clean.csv", 'a+', encoding='utf-8', newline=""))
    # list_writer.writerow(["img_name", "subject_id", "study_id", "finding", "impression"])
    list_writer.writerow(["subject_id", "study_id", "findings", "impression"])

    for index, row in tqdm(new_test_data.iterrows()):
        subject_id = row["subject_id"]
        study_id = row["study_id"]
        findings = row["findings"]
        impression = row["impression"]

        findings = findings.replace('\r', '')
        findings = findings.replace('\t', '')

        impression = impression.replace('\r', '')
        impression = impression.replace('\t', '')

        """delete note like 'talk to Doc, at pm, where'"""
        sentence_list = impression.split('.')
        # "with Dr", "by Dr", "to Dr",
        key_words = ["email", "phone", "Dr", "contact", "discuss", "minutes", "review", "dictation", "observation",
                     "communi"]

        first_cut_pos = 0
        temp_key = []
        for sentence_index, single_sentence in enumerate(sentence_list):
            for keyy in key_words:
                if keyy in single_sentence:
                    temp_key.append(keyy)
                    # find_pos = single_sentence.find(keyy)
                    # first_cut_pos += find_pos
                    break
                # else:
                #     if len(temp_key) != 0:
                #         first_cut_pos += len(single_sentence)
            if len(temp_key) != 0:
                break

        # if sentence_index == 0:
        #     if len(temp_key) == 0:
        #         first_cut_pos = len(impression)
        #     else:
        #         first_cut_pos = find_pos
        # else:
        #     for ii in range(sentence_index):
        #         first_cut_pos += (len(sentence_list[ii]) + 1)
        #
        # new_imp_text = impression[:first_cut_pos]

        if len(temp_key) == 0:
            new_imp_text = impression
        else:
            for ii in range(sentence_index):
                first_cut_pos += (len(sentence_list[ii]) + 1)
            new_imp_text = impression[:first_cut_pos]

        # print(index, "\n", new_imp_text)
        if len(new_imp_text.split()) < 2:
            print(index, "\n", new_imp_text)
            continue
        elif len(findings.split()) < 11:
            continue
        else:
            list_writer.writerow([subject_id, study_id, findings, new_imp_text])



if __name__ == '__main__':
    # extract_sections()
    # select_test_data_from_sections()

    data_clean_for_mimic()

