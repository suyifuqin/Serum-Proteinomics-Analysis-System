# 生成全局特征重要性图
plt.figure(figsize=(12, 8))
shap_values = explainer.shap_values(traindata)

# 使用SHAP的summary_plot函数
shap.summary_plot(
    shap_values,
    traindata,
    feature_names=MODEL_CONFIG[model_name]['features'],
    show=False,
    plot_size=(12, 8)
)
plt.title('Global Feature Importance', fontsize=14)
plt.tight_layout()
summary_plot_path = f"static/summary_plot_{int(time.time())}.png"
plt.savefig(summary_plot_path)
plt.close()