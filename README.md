# introduce

The web dynamically integrates serum proteomic inputs to deliver phase-specific risk assessments, visualized trajectory mapping, and biomarker-driven intervention protocols—transforming complex molecular data into precision decision support for radiation injury management."

## Installation

```python
pip install flask==3.0
pip install shap==0.48.0
pip install scipy==1.15.3
pip install numpy==2.2.6
pip install pandas==2.3.0
pip install joblib==1.5.1
pip install scikit-learn==1.6.1
```

# interface

Now, I will briefly introduce how to use the interface.

## index

This is the system's index page. Here, you can select five models - Multilayer Perceptron (MLP), Support Vector Machine (SVM), Random Forest (RF), Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM).

![image](docs/images/index.jpeg)

## input

After selecting the model, here you can input the genes of the protein:  Gene Input-Sell, Ltf, Masp1, Bpifa2 and lgfals

![image](docs/images/input.jpeg)

## output

The protein gene input was processed through the model prediction, resulting in the prediction results. It is also possible to observe the significance of different genes in the outcome. These results are presented through the SHAP Decision Plot and the SHAP Waterfall Plot.

![image](docs/images/output.jpeg)