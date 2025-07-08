from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

np.random.seed(42)

def cross_val_with_fixed_seed(model, X_train, y_train):
    # 固定随机种子的交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()


def optimize_xgb(X_train, y_train):
    print("定义xgboost模型，参数寻优...")

    # 定义目标函数，用于交叉验证评估模型性能
    def xgb_cv(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda,
               subsample, colsample_bytree, min_child_weight, gamma):
        model = XGBClassifier(
            n_estimators=int(n_estimators),  # 树的数量（迭代次数），整数
            max_depth=int(max_depth),  # 树的最大深度，控制模型复杂度，整数
            learning_rate=learning_rate,  # 学习率（步长）
            reg_alpha=reg_alpha,  # L1 正则化（权重的绝对值惩罚项）
            reg_lambda=reg_lambda,  # L2 正则化（权重的平方惩罚项）
            subsample=subsample,  # 每棵树使用的样本比例
            colsample_bytree=colsample_bytree,  # 每棵树使用的特征比例
            min_child_weight=min_child_weight,  # 子节点的最小样本权重总和
            gamma=gamma,  # 分裂所需的最小损失减少
            random_state=42,  # 随机数种子，确保结果可重复
            seed=42  # 兼容旧版本的随机数种子参数
        )
        return cross_val_with_fixed_seed(model, X_train, y_train)  # 使用自定义交叉验证函数评估模型性能

    # 定义参数搜索空间
    bo = BayesianOptimization(
        xgb_cv,  # 目标函数
        {
            'n_estimators': (10, 100),  # 树的数量
            'max_depth': (3, 15),  # 树的最大深度
            'learning_rate': (0.01, 0.3),  # 学习率
            'reg_alpha': (0.1, 1),  # L1 正则化
            'reg_lambda': (0.1, 1),  # L2 正则化
            'subsample': (0.5, 1.0),  # 样本采样比例（范围 [0.5, 1.0]）,控制每棵树随机采样的比例，较小的值可以防止过拟合
            'colsample_bytree': (0.5, 1.0),  # 特征采样比例（范围 [0.5, 1.0]）
            'min_child_weight': (2, 10),  # 子节点的最小样本权重总和（范围 [1, 10]）,较高的值可以防止过拟合
            'gamma': (0, 5),  # 节点分裂所需的最小损失减少值（范围 [0, 5]）
        },
        random_state=42
    )

    # 启动贝叶斯优化
    bo.maximize(init_points=3, n_iter=20)  # 初始采样点数=3，优化迭代次数=20
    # 获取最佳参数
    params = bo.max['params']
    print(params)

    # 使用最佳参数重新训练模型
    model = XGBClassifier(
        n_estimators=int(params['n_estimators']),  # 转为整数
        max_depth=int(params['max_depth']),  # 转为整数
        learning_rate=params['learning_rate'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        random_state=42,
        seed=42
    )
    return model


def optimize_lgbm(X_train, y_train):
    print("定义 LightGBM 模型，进行参数优化...")
    # 定义目标函数
    def lgb_cv(n_estimators, min_child_samples, feature_fraction, max_depth, learning_rate, reg_alpha, reg_lambda):
        model = LGBMClassifier(
            n_estimators=int(n_estimators),
            min_child_samples=int(min_child_samples),  # 替换为正确参数
            feature_fraction=min(feature_fraction, 0.999),  # 替换为正确参数
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=-1
        )
        return cross_val_with_fixed_seed(model, X_train, y_train)

    # 定义参数范围
    bo = BayesianOptimization(
        lgb_cv,
        {
            'n_estimators': (20, 200),  # 较少的树数量，防止过拟合
            'min_child_samples': (2, 10),  # 限制叶子节点的最小样本数，控制复杂度
            'feature_fraction': (0.7, 1.0),  # 高特征采样比例，确保大部分特征被使用
            'max_depth': (3, 7),  # 限制树的深度，降低过拟合风险
            'learning_rate': (0.01, 0.1),  # 较低的学习率，保证平稳学习
            'reg_alpha': (0, 1),  # 较大的 L1 正则化范围
            'reg_lambda': (0, 1),  # 较大的 L2 正则化范围
        },
        random_state=42
    )

    # 开始优化
    bo.maximize(init_points=3, n_iter=10)

    # 获取最佳参数
    params = bo.max['params']
    model = LGBMClassifier(
        n_estimators=int(params['n_estimators']),
        min_child_samples=int(params['min_child_samples']),  # 替换为正确参数
        feature_fraction=min(params['feature_fraction'], 0.999),  # 替换为正确参数
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42,
        verbosity=-1
    )

    return model


def optimize_svm(X_train, y_train):
    print("定义svm模型，参数寻优...")
    def svm_cv(C, gamma):
        model = SVC(C=C, gamma=gamma,random_state=42)
        return cross_val_with_fixed_seed(model, X_train, y_train)

    bo = BayesianOptimization(svm_cv, {'C': (1, 1000), 'gamma': (0.000001, 0.0001)}, random_state=42)
    bo.maximize(init_points=3, n_iter=20)
    params = bo.max['params']
    model = SVC(**params, kernel='rbf', probability=True, random_state=42)
    return model


def define_mlp():
    print("定义mlp模型,...")
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='logistic', #{'identity', 'logistic', 'tanh', 'relu'}
        solver='adam',
        max_iter=5000,
        alpha=0.001,
        random_state=42,
        verbose=False,
        tol=1e-5,
        early_stopping=False
    )


def optimize_rf(X_train, y_train):
    print("定义随机森林模型，参数寻优...")
    def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
        rf_model = RandomForestClassifier(n_estimators=int(n_estimators),
                                          min_samples_split=int(min_samples_split),
                                          max_features=min(max_features, 0.999),  # float
                                          max_depth=int(max_depth),
                                          random_state=42)
        return cross_val_with_fixed_seed(rf_model, X_train, y_train)

    rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': [100, 400],
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (5, 20),
         },
        random_state=42
    )

    rf_bo.maximize(init_points=3, n_iter=10)
    print(rf_bo.max)  # 可以通过该属性访问找到的参数和目标值的最佳组合
    para = rf_bo.max['params']
    n_estimators = para['n_estimators']
    min_samples_split = para['min_samples_split']
    max_features = para['max_features']
    max_depth = para['max_depth']

    rf_model = RandomForestClassifier(n_estimators=int(n_estimators),
                                      min_samples_split=int(min_samples_split),
                                      max_features=min(max_features, 0.999),  # float
                                      max_depth=int(max_depth),
                                      class_weight='balanced',
                                      random_state=42)
    return rf_model