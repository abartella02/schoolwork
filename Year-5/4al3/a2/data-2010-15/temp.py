import numpy as np
import os
from pathlib import Path
import json
data = {}

files = os.scandir(Path(__name__).parent)

for file in files:
    if file.is_file() and file.name.endswith(".npy"):
        data[file.name.removesuffix('.npy')] = np.load(file.name, allow_pickle=True)


for key in data.keys():
    print(f"Name: {key}, Len: {len(data[key])}, Size: {data[key].size}")

for key in data.keys():
    print(f"\n\n*************************\n{key}")
    print(f"Len: {len(data[key])}, Size: {data[key].size}")
    print(data[key])


a = data["neg_features_historical"]
a = 1

def feature_creation(fs_value: str):
    if fs_value == "FS-I":
        pos = data['pos_features_main_timechange'][:, :18]
        neg = data['neg_features_main_timechange'][:, :18]
        pos_cls = []
        neg_cls = []
        for idx, _ in enumerate(pos):
            pos_cls.append(data['pos_class'][idx][2])  # get magnitude
        for idx, _ in enumerate(neg):
            neg_cls.append(data['neg_class'][idx][2])

        return np.vstack((data['pos_features_main_timechange'][:, :18], data['neg_features_main_timechange'][:, :18]))

    if fs_value == "FS-II":
        return np.vstack((data['pos_features_main_timechange'][:, 19:], data['neg_features_main_timechange'][:, 19:]))

    if fs_value == "FS-III":
        return np.vstack((data['pos_features_historical'], data['neg_features_historical']))

    if fs_value == 'FS-IV':
        return np.vstack((data['pos_features_maxmin'], data['neg_features_maxmin']))





with open("all_harps_with_noaa_ars.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()

key = []
for line in lines:
    if not ("HARPNUM" in line or "NOAA_ARS" in line):
        harpnum, noaa_ars = line.strip().split()
        key.append((int(harpnum), [int(el.strip()) for el in noaa_ars.split(',')]))

print(key)

def get_pair(hnum = None, noaa = None):
    if hnum and noaa:
        return None
    if hnum:
        for k in key:
            if k[0] == int(hnum):
                return k[1]
    if noaa:
        for k in key:
            if any(n == int(noaa) for n in k[1]):
                return k[0]
"""
classification = []
for idx, i in enumerate(data['goes_data']):
    noaa = i['noaa_active_region']
    hnum = get_pair(noaa=noaa)
    if hnum:
        hnum = int(hnum)
        for jdx, j in enumerate(data['neg_class']):
            if int(j[0]) == int(hnum):
                classification.append((idx, jdx, hnum, noaa, i['goes_class']))
"""

"""
if norm_data.ndim > 1:
    for rownum, row in enumerate(norm_data):
        if any(np.isnan(x) for x in row):
            imputed_data = np.delete(norm_data, (rownum), axis=1)
else:
    for idx, el in enumerate(norm_data):
        if np.isnan(el):
            imputed_data = np.delete(norm_data, (idx), axis=0)
"""


"""

        bad_datapoints = []
        for key, data in self.data_norm.items():
            if key != "data_order":
                for rowNum, row in enumerate(data):
                    for colNum, el in enumerate(
                        row.items() if isinstance(row, dict) else row
                    ):
                        if (isinstance(el, str) and not el) or (
                            isinstance(el, np.floating) and np.isnan(el)
                        ):
                            if rowNum not in bad_datapoints:
                                bad_datapoints.append(rowNum)



"""

a = data['neg_class']
b = data['goes_data']

a = 1


