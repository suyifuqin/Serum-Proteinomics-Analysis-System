# -*- coding: utf-8 -*-
# @Time : 2024-10-15 14:38
import pandas as pd
import numpy as np
import os
from scipy import stats


f1 = r'../../../热图/早期/3d_子集_heatmap_proteins.csv'
f2 = r'../../../热图/急期/7d_交集添加后_heatmap_proteins.csv'


df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

protein_list1 = df1.values.flatten().tolist()
protein_list1 = [protein for protein in protein_list1 if pd.notna(protein)]

protein_list2 = df2.values.flatten().tolist()
protein_list2 = [protein for protein in protein_list2 if pd.notna(protein)]

# 将列表转换为集合
set1 = set(protein_list1)
set2 = set(protein_list2)

# 找到两个集合的交集
intersection = set1.intersection(set2)

# 将交集转换回列表
intersection_list = list(intersection)

# 打印结果
print("Intersection of the two lists:")
print(intersection_list)

intersection_df = pd.DataFrame(intersection_list, columns=['pro_name'])
intersection_df.to_csv('model2.csv', index=False)
