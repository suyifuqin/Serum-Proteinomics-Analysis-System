# -*- coding: utf-8 -*-
# @Time : 2024-10-28 10:55
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')  # 使用交互式后端
import seaborn as sns
import joblib
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from matplotlib import rcParams
from models import optimize_xgb, optimize_lgbm, optimize_svm, define_mlp,optimize_rf
from sklearn import metrics
from scipy.interpolate import CubicSpline
from scipy import interpolate
from imblearn.over_sampling import SMOTE
from  scipy.ndimage  import  gaussian_filter1d


# 设置保存路径
# save_path = rf"../../建模/分类结果_复现/低剂量/晚/25p_plot修改"
save_path = rf"../../建模/分类结果_复现_0703/急期5p_lgbm/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 读取数据
# data = pd.read_csv(r'../../建模/分类数据/traindata/低剂量/6h_1d_31p_rf.csv')  # 替换为你的文件路径
# data = pd.read_csv(r'../../建模/分类数据/traindata/恢复期/7d_11d_14d_10p_rf.csv')  # 替换为你的文件路径
data = pd.read_csv(r'../../建模/分类数据/traindata/急期/16p_lgbm.csv')  # 替换为你的文件路径
# data = pd.read_csv(r'../../建模/分类数据/traindata/早期/5p.csv')  # 替换为你的文件路径
data = data.head(16-11)

# class_labels = ["正常","干预","亚致死","致死"]  #早期和急期
# class_labels = ["非致死","致死"]  #急期
# class_labels = ["恢复良好","恢复差"]  #预后，恢复期
# class_labels = ["正常","低剂量","干预"]  #低剂量

class_labels = ['Untreatment','Treatment','Sublethal','Lethal']
# class_labels = ['Non-lethal','Lethal']
# class_labels = ['Good prognosis','Poor prognosis']
# class_labels = ['Normal','LDR','Treatment']
# class_labels = ['Others','LDR']

n_classes = len(class_labels)

# 提取蛋白名称（第一列）作为特征名称
protein_names = data.iloc[:, 0].values
# 筛选样本列名：仅保留包含 "Gy" 的列
sample_columns = [col for col in data.columns if "Gy" in col]
samples_data = data[sample_columns]

# 提取辐射量信息作为分类依据
radiation_levels = [col.split('_')[0] for col in sample_columns]

if n_classes==4:
    # 早期，急期
    radiation_levels = [
        'Low' if level in ['0Gy', '0.2Gy','0.5Gy']  else level for level in radiation_levels
    ]
    # k=5  #早期
    k=3   #急期
elif n_classes==2:
    # 预后
    radiation_levels = [
        # 'Non-lethal' if level in ['0Gy','0.2Gy','0.5Gy','2Gy','6.5Gy']  else level for level in radiation_levels
        'Others' if level in ['0Gy','0.5Gy','2Gy']  else level for level in radiation_levels
        # '预后良好' if level in ['0Gy', '2Gy','6.5Gy+A','10Gy+A'] else level for level in radiation_levels
    ]
    k=3
elif n_classes==3:
    # 低剂量
    radiation_levels = [
        'Normal' if level in ['0Gy'] else
        'LDR' if level in ['0.2Gy','0.5Gy'] else
        'Treatment' if level in ['2Gy'] else level
        for level in radiation_levels
    ]
    k=3
else:
    print("未定义的类别！！！")


radiation_labels = pd.Series(radiation_levels).factorize()[0]

# 构造特征矩阵和标签
X = samples_data.T  # 转置，使每列为一个样本
y = radiation_labels

# 将蛋白质名称映射为列名
X.columns = protein_names

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# 应用 SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42,k_neighbors=k)
X_train, y_train = smote.fit_resample(X_train, y_train)


# 特征缩放
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# 恢复 DataFrame 格式并保留蛋白质名称
X_train = pd.DataFrame(X_train_scaler, columns=protein_names)
X_test = pd.DataFrame(X_test_scaler, columns=protein_names)

# 定义模型
model_xgb = optimize_xgb(X_train, y_train)
model_lgbm = optimize_lgbm(X_train, y_train)
model_svm = optimize_svm(X_train, y_train)
model_mlp = define_mlp()
model_rf = optimize_rf(X_train, y_train)


models = [model_mlp,model_svm,model_xgb,model_lgbm,model_rf]
models_name = ['MLP model', 'SVM model', 'XGBoost model', 'LightGBM model','RF model']
# models = [model_lgbm]
# models_name = ['LightGBM model']

mean_fpr = np.linspace(0, 1, 21)
# 将目标标签二值化，用于多分类计算
y_test_bin = label_binarize(y_test, classes=range(n_classes))
# 配置中文字体（以 SimHei 为例，适用于 Windows 系统）
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用黑体SimHei # 雅黑Microsoft YaHei
rcParams['axes.unicode_minus'] = False  # 避免负号显示问题
rcParams['font.family'] = 'Arial'

# 绘制每个模型的所有类别 ROC 曲线
def plot_individual_model_roc_class2(model_name, y_pro):
    auc_score = metrics.roc_auc_score(y_test, y_pro[:, 1])
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pro[:, 1])
    idx = [t - f for t, f in zip(tpr, fpr)]
    best_threshold_idx = idx.index(max(idx))
    best_threshold = thresholds[best_threshold_idx]

    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    model_tprs.append(mean_tpr)
    model_aucs.append(auc_score)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=3)

    plt.plot(fpr, tpr, lw=2, label=f'{class_labels[1]} (AUC = {auc_score:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=21)
    plt.ylabel('True Positive Rate', fontsize=21)
    plt.title(f'{model_name} ROC Curve', fontsize=21)
    plt.legend(loc="lower right",fontsize=18)
    # 修改横纵坐标字体大小
    plt.tick_params(axis='x', labelsize=20)  # 修改x轴字体大小
    plt.tick_params(axis='y', labelsize=20)  # 修改y轴字体大小
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f"{save_path}/ROC_{model_name}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    return model_tprs, model_aucs, best_threshold


def plot_individual_model_roc(model_name, y_pro):
    tprs = []
    aucs = []
    ori_fprs = []
    ori_tprs = []
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=3)

    for i, label in enumerate(class_labels):
        fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_pro[:, i])
        # auc_score = auc(fpr, tpr)
        auc_score = metrics.roc_auc_score(y_test_bin[:, i], y_pro[:, i])

        mean_tpr = np.interp(mean_fpr, fpr, tpr)

        tprs.append(mean_tpr)
        aucs.append(auc_score) #保存AUC

        ori_fprs.append(fpr)
        ori_tprs.append(tpr)

        plt.plot(mean_fpr, mean_tpr, lw=3, label=f'{label} (AUC = {auc_score:.4f})')
        # plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc_score:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=21)
    plt.ylabel('True Positive Rate', fontsize=21)
    plt.title(f'{model_name} ROC Curve', fontsize=21)
    plt.legend(loc="lower right",fontsize=18)
    # 修改横纵坐标字体大小
    plt.tick_params(axis='x', labelsize=20)  # 修改x轴字体大小
    plt.tick_params(axis='y', labelsize=20)  # 修改y轴字体大小
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f"{save_path}/ROC_{model_name}.png", bbox_inches='tight')
    # plt.show()
    plt.close()
    return tprs, aucs

# 绘制所有模型的综合 ROC 曲线
def plot_all_models_roc(all_model_tprs, fprs, all_model_aucs):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
    color_all = ['black', 'red', 'green', 'yellow', 'blue', 'gold']

    for i, model_name in enumerate(models_name):
        # 计算每个模型的平均 TPR 和宏平均 AUC
        mean_tpr = np.mean(all_model_tprs[i], axis=0)  # 按类别取平均 TPR
        mean_tpr[-1] = 1.0  # 确保最后点为1
        mean_auc = np.mean(all_model_aucs[i])  # 计算宏平均 AUC
        # print(all_model_aucs[i])
        # print(mean_auc)

        # 绘制模型的平均 ROC 曲线
        plt.plot(
            fprs, mean_tpr,
            color=color_all[i],
            lw=3,
            label=f'{model_name} (Mean AUC = {mean_auc:.4f})',
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=21)
    plt.ylabel('True Positive Rate', fontsize=21)
    plt.title('All Models Mean ROC Curve', fontsize=21)
    plt.legend(loc="lower right",fontsize=16)
    # 修改横纵坐标字体大小
    plt.tick_params(axis='x', labelsize=20)  # 修改x轴字体大小
    plt.tick_params(axis='y', labelsize=20)  # 修改y轴字体大小
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f"{save_path}/ROC_all_models_mean.png",  bbox_inches='tight') #dpi=300
    plt.close()


def plot_confusion_matrix(conf_matrix, model_name, labels, normalize=False):
    # 归一化
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6.5))
    # plt.figure(figsize=(7, 5.7)) #分两类
    hm = sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 24, "color": "black"},  # 修改数值字体 20
                cbar_kws={"shrink": 1, "aspect": 20} #分两类时修改画图比例aspect 10
                )
    plt.xlabel('Predicted Label',fontsize=25)  # 21
    plt.ylabel('True Label', fontsize=25)  # 21
    plt.title(f'{model_name} Confusion Matrix', fontsize=25)  # 21

    # 修改数值条字体大小
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    # 修改横纵坐标字体大小
    plt.tick_params(axis='x', labelsize=20, rotation=0)  # 修改x轴字体大小
    plt.tick_params(axis='y', labelsize=20, rotation=80)  # 修改y轴字体大小
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(save_path,f"{model_name}_混淆矩阵.png"))
    plt.close()


# 存储所有模型的 TPR 和 AUC
all_model_tprs = []
all_model_aucs = []
models_trained = []
for model, model_name in zip(models, models_name):
    print(f"模型训练和评估，{model_name}...")
    # 分类报告保存路径
    report_path = os.path.join(save_path, f"{model_name}_classification_report.txt")
    report_file = open(report_path, "w", encoding="utf-8")

    # 1. 交叉验证（基于训练集）
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=cv)

    # 写入交叉验证结果
    report_file.write(f"模型: {model_name}\n")
    report_file.write(f"交叉验证平均准确率: {cross_val_scores.mean():.4f}\n")
    report_file.write(f"交叉验证标准差: {cross_val_scores.std():.4f}\n")

    # 混淆矩阵（基于训练集交叉验证）
    cm_cv = confusion_matrix(y_train, y_cv_pred, labels=range(n_classes))
    report_file.write("交叉验证混淆矩阵:\n")
    report_file.write(np.array2string(cm_cv, separator=",") + "\n")

    # 分类报告（基于训练集交叉验证）
    class_report_cv = classification_report(y_train, y_cv_pred, target_names=class_labels)
    report_file.write("交叉验证分类报告:\n")
    report_file.write(class_report_cv + "\n")
    report_file.write("-" * 80 + "\n")

    # 2. 训练最终模型（基于整个训练集）
    model.fit(X_train, y_train)

    # 3. 验证集模型预测
    y_pro = model.predict_proba(X_test)  # shape: [n_samples, n_classes]
    y_test_pred = np.argmax(y_pro, axis=1)  # 初步预测（最大概率类别）

    # 4. 画ROC曲线
    model_tprs = []
    model_aucs = []
    if len(class_labels) == 2:
        model_tprs, model_aucs, best_threshold = plot_individual_model_roc_class2(model_name, y_pro)
    else:
        model_tprs, model_aucs = plot_individual_model_roc(model_name, y_pro)
    # 存储结果以绘制综合 ROC 图
    all_model_tprs.append(model_tprs)
    all_model_aucs.append(model_aucs)

    # 5. 计算混淆矩阵和精度
    # if n_classes==2:
    #     # 根据最佳阈值调整输出
    #     y_test_pred_adjusted = np.copy(y_test_pred)
    #     for i in range(len(y_pro)):
    #         if y_pro[i,1] >= best_threshold:
    #             y_test_pred_adjusted[i] = 1
    #         else:
    #             y_test_pred_adjusted[i] = 0
    #     y_test_pred = y_test_pred_adjusted

    # 计算混淆矩阵
    cm_test = confusion_matrix(y_test, y_test_pred, labels=range(n_classes))
    report_file.write("测试集混淆矩阵:\n")
    report_file.write(np.array2string(cm_test, separator=",") + "\n")
    # 绘制混淆矩阵
    plot_confusion_matrix(cm_test, model_name, class_labels, normalize=False)

    class_report_test = classification_report(y_test, y_test_pred, target_names=class_labels)
    report_file.write("测试集分类报告:\n")
    report_file.write(class_report_test + "\n")
    report_file.write("-" * 80 + "\n")
    if len(class_labels) == 2:
        report_file.write("最佳阈值：\n")
        report_file.write(str(best_threshold))
    report_file.close()


    # 6. 保存模型
    joblib.dump((model,scaler), f'{save_path}/{model_name}.pkl')
    models_trained.append(model)


# 绘制所有模型的综合 ROC 曲线
print("画ROC曲线...")
plot_all_models_roc(all_model_tprs, mean_fpr, all_model_aucs)


np.random.seed(42)
# SHAP解释
for model, model_name in zip(models_trained, models_name):
    print(f"SHAP 分析，{model_name}...\n")

    np.random.seed(42)
    if model_name in ['XGBoost model', 'LightGBM model', 'RF model']:
        explainer = shap.TreeExplainer(model)
    elif model_name in ['MLP model', 'SVM model']:
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
    else:
        explainer = shap.Explainer(model, X_train)

    # 返回所有类别的 shap 值
    shap_values_all = explainer.shap_values(X_test)

    # 计算平均绝对 SHAP 值
    if shap_values_all.ndim>2:
        shap_values = np.mean(np.abs(shap_values_all.T), axis=0).T
    else:
        shap_values = shap_values_all

    # 绘制特征重要性图（条形图）
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=protein_names,show=False)
    # shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=protein_names,show=False,plot_size=(9.5,10))
    plt.title(f'{model_name} Summary Plot Bar', fontsize=23) #20
    plt.xticks(fontsize=22)  # x轴标签 20
    plt.yticks(fontsize=24)  # y轴标签 20
    # plt.gca().xaxis.label.set_size(22)  # 修改x轴标签字体大小
    plt.xlabel("Mean(|SHAP value|)", fontsize=22)
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(save_path, f"{model_name}_summary_plot_bar.png"))
    plt.close()

    # 绘制 SHAP 值的摘要图
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", feature_names=protein_names,show=False)
    # shap.summary_plot(shap_values, X_test, plot_type="dot", feature_names=protein_names,show=False,plot_size=(9.5,10))
    plt.title(f'{model_name} Summary Plot Dot', fontsize=20)
    plt.xticks(fontsize=20)  # x轴标签
    plt.yticks(fontsize=20)  # y轴标签
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(save_path, f"{model_name}_summary_plot_dot.png"),bbox_inches='tight')
    plt.close()

    # # 为每个类别绘制 SHAP Summary Plot
    # for i in range(n_classes):
    #     plt.figure(figsize=(8, 6))
    #     shap.summary_plot(shap_values=shap_values_all.T[i].T,  features=X_test, feature_names=protein_names,
    #                       plot_type="dot", show=False)
    #     plt.savefig(save_path + f"{model_name}_summary_plot_{class_labels[i]}.png", bbox_inches="tight")
    #     plt.savefig(os.path.join(save_path, f"{model_name}_summary_plot_{class_labels[i]}.png"), bbox_inches="tight")
    #     plt.close()

    # 遍历每个类别，绘制 decision plot
    if shap_values_all.ndim > 2:
        for i in range(n_classes):
            plt.figure(figsize=(9.5, 10))
            shap.decision_plot(
                explainer.expected_value[i],
                shap_values_all.T[i].T,
                X_test,
                feature_names=protein_names,
                show=False,
                auto_size_plot=True   ###### 修改
            )
            plt.title(f'{model_name} Decision for Class {class_labels[i]}', fontsize=22)
            plt.xticks(fontsize=22)  # x轴标签 20
            plt.yticks(fontsize=23)  # y轴标签 20
            plt.xlabel("Model output value", fontsize=22)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{model_name}_decision_plot_class_{class_labels[i]}.png"))
            plt.close()
    else:
        plt.figure(figsize=(9.5, 10))
        shap.decision_plot(
            explainer.expected_value,
            shap_values_all,
            X_test,
            feature_names=protein_names,
            show=False,
            auto_size_plot=False
        )
        plt.title(f'{model_name} Decision for Class {class_labels[1]}', fontsize=22)
        plt.xticks(fontsize=22)  # x轴标签 20
        plt.yticks(fontsize=23)  # y轴标签 20
        plt.xlabel("Model output value", fontsize=22)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{model_name}_decision_plot_class_{class_labels[1]}.png"))
        plt.close()


    # 按shap排序保存结果
    # shap_importance = np.abs(shap_values).mean(axis=0)
    # shap_df = pd.DataFrame({
    #     'Feature': protein_names,
    #     'SHAP Importance': shap_importance
    # })
    # shap_df_sorted = shap_df.sort_values(by='SHAP Importance', ascending=False)
    # sorted_features = shap_df_sorted['Feature'].tolist()
    # data_sorted = data.set_index(data.columns[0])
    # data_sorted = data_sorted.loc[sorted_features]
    # data_sorted.reset_index(inplace=True)
    # data_sorted.to_csv(rf'{save_path}/{model_name}_sorted.csv', index=False)
    #
    # shap_df_sorted['Percentage'] = shap_df_sorted['SHAP Importance'] / shap_df_sorted['SHAP Importance'].sum()
    # shap_df_sorted['Cumulative Percentage'] = shap_df_sorted['Percentage'].cumsum()
    # shap_df_sorted.to_csv(rf'{save_path}/{model_name}_shapvalue.csv',index=False)
