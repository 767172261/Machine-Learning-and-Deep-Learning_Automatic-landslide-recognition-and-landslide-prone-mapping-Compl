机器学习和深度学习的_滑坡自动识别和滑坡易发性制图 基于滑坡自动识别和滑坡易发性制图的完整实现
@[toc]
以下文字及代码仅供参考。

滑坡自动识别和滑坡易发性制图![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5ec9b695f6b14051a17eeb87d4989d1a.png)

机器学习和深度学习实现
语言：Python
软件：Arcgis，Jupyter notebook或Pycharm
模型：随机森林，支持向量机，神经网络
过程：1. 数据集制作（自己准备影像和滑坡标注，本人提供影像是有偿的）2. 代码（数据读取、数据划分、模型训练、应用模型、绘制ROC曲线、精度评价、预测结果输出）3. 结果制图。

基于滑坡自动识别和滑坡易发性制图的完整实现流程，使用 Python 和 ArcGIS、Jupyter Notebook 或 PyCharm 环境。我数据集制作、代码实现（包括随机森林、支持向量机和神经网络模型）、结果制图等步骤。

---

### **1. 数据集制作**

#### 1.1 数据来源
- **影像数据**：卫星影像（如 Sentinel-2、Landsat）或高分辨率 DEM 数据。
- **滑坡标注数据**：手动标注滑坡区域（可使用 ArcGIS 或 QGIS 标注）。

#### 1.2 数据预处理
- 使用 ArcGIS 或 QGIS 提取以下特征：
  - 地形因子：坡度、坡向、高程变化率等。
  - 植被指数：NDVI、NDWI。
  - 水文因子：河流密度、流域面积。
  - 土壤因子：土壤类型、渗透性。
  - 地质因子：岩性、断层距离。

#### 1.3 数据格式
- 将影像和标注数据导出为栅格文件（如 `.tif`）。
- 标注数据应为二值化标签（0 表示非滑坡，1 表示滑坡）。

---

### **2. 代码实现**

#### 2.1 数据读取与处理

```python
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# 读取影像数据
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
        profile = src.profile
    return data, profile

# 加载数据
image, profile = read_raster('path/to/image.tif')
labels, _ = read_raster('path/to/labels.tif')

# 展平数据
image = image.reshape(image.shape[0], -1).T
labels = labels.flatten()

# 去除无效值（如 NaN）
valid_mask = ~np.isnan(labels)
X = image[valid_mask]
y = labels[valid_mask]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

#### 2.2 模型训练

##### 随机森林模型

```python
# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 预测
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
```

##### 支持向量机模型

```python
# 训练支持向量机模型
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 预测
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
```

##### 神经网络模型

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 构建神经网络模型
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 预测
y_prob_nn = nn_model.predict(X_test).flatten()
y_pred_nn = (y_prob_nn > 0.5).astype(int)
```

---

#### 2.3 绘制 ROC 曲线

```python
# 计算 ROC 曲线和 AUC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
auc_svm = auc(fpr_svm, tpr_svm)

fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)
auc_nn = auc(fpr_nn, tpr_nn)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {auc_nn:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

#### 2.4 精度评价

```python
# 计算精度
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print(f'SVM Accuracy: {accuracy_svm:.2f}')
print(f'Neural Network Accuracy: {accuracy_nn:.2f}')
```

---

#### 2.5 预测结果输出

```python
# 使用随机森林模型进行预测
predicted_labels = rf_model.predict(image)

# 将预测结果保存为栅格文件
output_profile = profile.copy()
output_profile.update(dtype=rasterio.float32, count=1)
with rasterio.open('path/to/predicted_labels.tif', 'w', **output_profile) as dst:
    dst.write(predicted_labels.reshape(profile['height'], profile['width']).astype(np.float32), 1)
```

---

### **3. 结果制图**

#### 3.1 在 ArcGIS 中加载结果
- 打开 ArcGIS Pro 或 QGIS。
- 加载预测结果文件（`predicted_labels.tif`）。
- 设置分类渲染（如滑坡区域为红色，非滑坡区域为绿色）。

#### 3.2 制作易发性地图
- 在 ArcGIS 中叠加地形、水文等要素。
- 使用预测概率生成滑坡易发性地图（高概率区域为高风险区）。

---

### **总结**

从数据准备到模型训练、推理和结果制图的完整流程。同学你可以根据需求选择合适的模型（随机森林、SVM 或神经网络），并调整超参数以优化性能。

仅供参考，我的同学们
