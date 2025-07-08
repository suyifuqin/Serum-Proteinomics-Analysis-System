from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
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
import shap
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score,StratifiedKFold,cross_val_predict
from imblearn.over_sampling import SMOTE
import pickle
from matplotlib import rcParams
import joblib
from sklearn.metrics import classification_report
import time
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = '你的安全密钥'


# 模型配置（包含类别名称）
MODEL_CONFIG = {
    'Early-7Gene': {
        'path': 'models/早期SVM-7Gene.pkl',
        'features': ['Serpina3n', 'Saa4', 'Hp', 'Saa1', 'Coro1a', 'Psmb8', 'Timp2'],
        'example': [4646170, 630474, 8956846, 194747, 64055, 69838, 31112],
        'classes': ['Normal', 'Treatment', 'Sublethal', 'Lethal'],
        'train_data':"./sampledata/早期/7p_svm.csv"
    },
    'Acute-5Gene': {
        'path': 'models/急期MLP-5Gene.pkl',
        'features': ['Sell', 'Ltf', 'Masp1', 'Bpifa2', 'lgfals'],
        'example': [8346, 3554, 159208, 51349, 523320],
        'classes': ['Normal', 'Treatment', 'Sublethal', 'Lethal'],
        'train_data': "./sampledata/急期/5p_mlp.csv"
    },
    'Prognosis-3Gene': {
        'path': 'models/预后RF-3Gene.pkl',
        'features': ['Sell', 'Thbs1', 'Pf4'],
        'example': [30000, 4500000, 2000000],
        'classes': ['Good prognosis', 'Poor prognosis'],
        'train_data': "./sampledata/恢复期/3p_rf.csv"
    },
    'Low-dose Early-9Gene': {
        'path': 'models/低剂量早期RF-9Gene.pkl',
        'features': ['Psmb10', 'Coro1a', 'Ngp',  'Psma6','lgfbp4', 'Ca1', 'Psmb8', 'Msn', 'Psma1'],
        'example': [10000, 12000, 13000, 45100, 40000, 80000, 35000, 8000, 9000],
        'classes': ['Normal', 'LDR', 'Treatment'],
        'train_data':"./sampledata/低剂量/9p_rf.csv"
    },
    'Low-dose Late-3Gene': {
        'path': 'models/低剂量晚期RF-3Gene.pkl',
        'features': [ 'GCAB','Gm5629', 'KV3A1'],
        'example': [24751412, 676437, 12740307], 
        'classes': ['Others', 'LDR'],
        'train_data':"./sampledata/低剂量/3p_rf.csv"
    }
}

# 模型缓存
loaded_models = {}

def load_model(model_name):
    """加载模型和标准化器"""
    if model_name not in loaded_models:
        model_path = MODEL_CONFIG[model_name]['path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        loaded_models[model_name] = joblib.load(model_path)
    return loaded_models[model_name]

def get_prediction(models, input_data):
    """获取预测结果"""
    model = models[0]
    scaler = models[1]
    X = np.array([input_data])
    X = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)[0]
    else:
        prediction = model.predict(X)[0]
        probabilities = [1.0 if i == prediction else 0.0 for i in range(len(MODEL_CONFIG[model]['classes']))]
    
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, probabilities


def clean_old_plots():
    # 清理超过1小时的图像文件
    now = time.time()
    for filename in glob.glob('static/*_plot_*.png'):
        if os.stat(filename).st_mtime < now - 360:
            os.remove(filename)
            

def get_decision_plot(model_name, models, input_data, predicted_class):
    clean_old_plots()
    """Generate SHAP decision plot and save as image"""
    model = models[0]
    scaler = models[1]
    X = np.array([input_data])
    X = scaler.transform(X)
    
    traindata = pd.read_csv(MODEL_CONFIG[model_name]['train_data'])
    sample_columns = [col for col in traindata.columns if "Gy" in col]
    traindata = traindata[sample_columns]
    traindata = traindata.T
    traindata = scaler.transform(traindata)
    
    if model_name in ['Prognosis-3Gene', 'Low-dose Early-9Gene', 'Low-dose Late-3Gene']:
        explainer = shap.TreeExplainer(model)
    elif model_name in ['Early-7Gene', 'Acute-5Gene']:
        explainer = shap.KernelExplainer(model.predict_proba, traindata)
    else:
        explainer = shap.Explainer(model, traindata)
        
    # Generate decision plot
    plt.figure(figsize=(10, 6))
    if len(explainer.expected_value) > 1:  # Multi-class case
        shap.decision_plot(
            explainer.expected_value[predicted_class],
            explainer.shap_values(X)[0].T[predicted_class],
            X,
            feature_names=MODEL_CONFIG[model_name]['features'],
            show=False
        )
    else:  # Binary classification case
        shap.decision_plot(
            explainer.expected_value,
            explainer.shap_values(X),
            X,
            feature_names=MODEL_CONFIG[model_name]['features'],
            show=False
        )
    plt.title(f'Decision Plot', fontsize=14)
    plt.tight_layout()
    decision_plot_path = f"static/decision_plot_{int(time.time())}.png"
    plt.savefig(decision_plot_path)
    plt.close()
    
    # Generate waterfall plot
    print(111111)
    plt.figure()
    if len(explainer.expected_value) > 1:  # Multi-class case
        shap_values = explainer.shap_values(X)[0].T[predicted_class]
        base_value = explainer.expected_value[predicted_class]
    else:  # Binary classification case
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value
    print(22222)
    print(shap_values)
    print(base_value)
    print(X)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X[0],
        feature_names=MODEL_CONFIG[model_name]['features']
    )

    shap.plots.waterfall(explanation, show=False)
    plt.title(f'Waterfall Plot', fontsize=14)
    plt.tight_layout()
    waterfall_plot_path = f"static/waterfall_plot_{int(time.time())}.png"
    plt.savefig(waterfall_plot_path)
    plt.close()
    
    return decision_plot_path, waterfall_plot_path


  

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_model = request.form.get('model')
    features = []
    example_values = []
    result = None
    error = None

    if selected_model and selected_model in MODEL_CONFIG:
        config = MODEL_CONFIG[selected_model]
        features = config['features']
        example_values = config['example']

        if request.method == 'POST':
            if 'use_example' in request.form:
                input_data = example_values
            else:
                try:
                    input_data = []
                    for feature in features:
                        value = request.form.get(feature, '').strip()
                        if not value:
                            raise ValueError(f"Please fill in all fields！")
                        try:
                            input_data.append(float(value))
                        except ValueError:
                            raise ValueError(f"Invalid value: '{feature}'")
                    
                    if any(val < 0 for val in input_data):
                        raise ValueError("All the values should be greater than 0. ")
                        
                except ValueError as e:
                    error = str(e)
                else:
                    try:
                        model = load_model(selected_model)
                        predicted_class, probabilities = get_prediction(model, input_data)
                        # 生成图表
                        decision_plot, waterfall_plot = get_decision_plot(
                            selected_model, model, input_data, predicted_class
                        )
                        result = {
                            'class': predicted_class,
                            'probabilities': probabilities,
                            'input_data': input_data,
                            'decision_plot': decision_plot,
                            'waterfall_plot': waterfall_plot
                    }
                    except Exception as e:
                        error = f"预测过程中发生错误: {str(e)}"

    feature_examples = list(zip(features, example_values)) if features and example_values else []

    return render_template('index.html',
                         models=MODEL_CONFIG.keys(),
                         selected_model=selected_model,
                         feature_examples=feature_examples,
                         result=result,
                         error=error,
                         model_config=MODEL_CONFIG)  # 关键修改：传递MODEL_CONFIG到模板

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get uploaded file
            file = request.files['file']
            if file:
                # Read Excel file
                df = pd.read_excel(file)
                
                # Get selected model
                selected_model = request.form.get('model')
                if selected_model not in MODEL_CONFIG:
                    return render_template('index.html',
                        error='Invalid model selection',
                        model_config=MODEL_CONFIG)
                
                # Get model and prediction results
                model = load_model(selected_model)
                input_data = preprocess_data(df, selected_model)
                predicted_class, probabilities = get_prediction(model, input_data)
                
                # Generate plots
                decision_plot, waterfall_plot = get_decision_plot(
                    selected_model, model, input_data, predicted_class
                )
                
                # Return prediction results and plot paths
                return render_template('index.html',
                    result={
                        'prediction': predicted_class,
                        'probabilities': probabilities.tolist(),
                        'decision_plot': decision_plot,
                        'waterfall_plot': waterfall_plot,
                        'model_config': MODEL_CONFIG
                    },
                    model_config=MODEL_CONFIG)
        except Exception as e:
            return render_template('index.html',
                error=str(e),
                model_config=MODEL_CONFIG)
    return render_template('index.html',
        error='Invalid request method',
        model_config=MODEL_CONFIG)

if __name__ == '__main__':
    clean_old_plots()
    app.run(debug=True)
