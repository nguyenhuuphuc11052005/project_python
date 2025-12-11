
Đồ án cuối kỳ môn Python cho khoa học dữ liệu

## 📑 Mục lục

1. [Bối cảnh & Mục tiêu](#-bối-cảnh--mục-tiêu)
2. [Tính năng nổi bật](#-tính-năng-nổi-bật)
3. [Cấu trúc dự án](#-cấu-trúc-dự-án)
4. [Cài đặt (Cơ bản)](#-cài-đặt)
5. [Hướng dẫn sử dụng (Pipeline)](#-hướng-dẫn-sử-dụng)
6. [Hướng dẫn cài đặt & Chạy trên Local (Chi tiết)](#-hướng-dẫn-cài-đặt--chạy-trên-local-máy-cá-nhân)
7. [Kết quả thực nghiệm & So sánh](#-kết-quả-thực-nghiệm--so-sánh-model-evaluation)
8. [Ghi chú cho Google Colab](#-ghi-chú-cho-google-colab)
9. [Hướng phát triển tiếp theo](#-hướng-phát-triển-tiếp-theo-roadmap)
10. [Tác giả](#-tác-giả)



# 🏥 Dự đoán Chi Phí Y Tế (Medical Cost Prediction)

Dự án Machine Learning End-to-End nhằm dự đoán chi phí y tế hằng năm (`annual_medical_cost`) dựa trên hồ sơ nhân khẩu học, sức khỏe và bảo hiểm của bệnh nhân. Dự án được xây dựng theo hướng đối tượng (OOP) với các module tái sử dụng cao.

## 🎯 Bối cảnh & Mục tiêu
Chi phí chăm sóc sức khỏe đang là gánh nặng lớn đối với nhiều cá nhân và tổ chức bảo hiểm. Dự án này được xây dựng nhằm giải quyết bài toán: **"Liệu có thể dự đoán chính xác chi phí y tế hằng năm dựa trên hồ sơ cá nhân?"**

Dữ liệu bao gồm 100.000 bản ghi với các nhóm thông tin:
* **Demographics:** Tuổi, giới tính, vùng miền.
* **Lifestyle:** Chỉ số BMI, hút thuốc, tập thể dục.
* **Medical History:** Tiền sử bệnh lý (tiểu đường, cao huyết áp...).
* **Insurance:** Loại gói bảo hiểm, hạn mức.

**Mục tiêu chính:** Xây dựng mô hình hồi quy (Regression) để dự đoán `annual_medical_cost`, giúp các công ty bảo hiểm đánh giá rủi ro và cá nhân hóa gói dịch vụ.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20Pandas%20%7C%20Seaborn-green)

## 🌟 Tính năng nổi bật

* **Pipeline Tiền xử lý dữ liệu mạnh mẽ (`preprocess.py`):**
    * Tự động xử lý giá trị thiếu (Mean, Median, Forward-fill).
    * Phát hiện và xử lý ngoại lai (Outliers) bằng IQR hoặc Isolation Forest.
    * Feature Engineering: Tách ngày tháng, One-Hot Encoding, Ordinal Encoding.
    * Chuẩn hóa dữ liệu (Scaling) để chống rò rỉ dữ liệu (Data Leakage).
* **Huấn luyện mô hình tự động (`model_trainer.py`):**
    * Hỗ trợ chạy đua (Race) giữa nhiều mô hình: Random Forest, XGBoost, Linear Regression, SVM...
    * Tự động tối ưu tham số (Hyperparameter Tuning) bằng Random Search.
    * Lưu trữ artifact trọn gói (Model + Scaler + Metrics).
* **Trực quan hóa dữ liệu (`visualize.py`):**
    * Hệ thống vẽ biểu đồ chuẩn hóa, dễ dàng so sánh hiệu suất mô hình.
    * Hỗ trợ vẽ Dashboard so sánh đa chỉ số (RMSE, MAE, R2).

## 📂 Cấu trúc dự án

```text
├── data/                       # Chứa file dữ liệu gốc (csv, xlsx)
├── results/                    # Kết quả đầu ra (Logs, Models, Charts)
│   ├── best_model.pkl          # Model tốt nhất đã huấn luyện
│   ├── training.log            # Log quá trình chạy
│   └── model_comparison.csv    # Bảng so sánh các model
│   └── các plot so sánh các model
│
├── preprocessing/              # Thư mục module chính
│   ├── __init__.py             # File khởi tạo module
│   ├── base.py                 # Chứa BasePreprocessor
│   ├── imputer.py              # Chứa class Imputer
│   ├── scaler.py               # Chứa class Scaler
│   ├── outlier_handler.py      # Chứa class OutlierHandler
│   ├── feature_engineer.py     # Chứa class FeatureEngineer
│   └── manager.py              # Chứa class DataManager
└── demo_preprocess.py          # File chạy demo cho module preprocessing
│
├── model_training/             # Folder Module
│   ├── __init__.py             # Khởi tạo module
│   ├── logger_config.py        # Cấu hình logging
│   └── trainer.py              # Chứa class ModelTrainer chính
└── demo_training.py            # File script để chạy demo module model_traning
│
├── visualize.py                # Module trực quan hóa
├── EDA.ipynb                   # Notebook để chạy phần EDA của dự án
├── FE_MODELING.ipynb           # Notebook để chạy feature engineering và modeling
├── requirements.txt            # Các thư viện cần thiết
└── README.md                   # Hướng dẫn sử dụng
````

## 🛠️ Cài đặt

### 1\. Yêu cầu hệ thống

  * Python 3.8 trở lên.
  * Các thư viện: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, xgboost, lightgbm.

### 2\. Cài đặt thư viện

Chạy lệnh sau để cài đặt các gói cần thiết:

```bash
pip install -r requirements.txt
```



-----

## 🚀 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu và Tiền xử lý

Sử dụng `DataManager` và các bộ xử lý trong module `preprocesing`.

```python
from preprocessing import DataManager, Imputer, Scaler, OutlierHandler, FeatureEngineer

# 1. Load dữ liệu
manager = DataManager('data/medical_cost.csv')

# 2. Định nghĩa Pipeline
steps = [
    Imputer(strategy='mean', columns=['bmi', 'income']),
    FeatureEngineer(one_hot_cols=['region', 'smoker'], ordinal_cols={'education': ['No HS', 'HS', 'Bachelor']}),
    OutlierHandler(method='isolation_forest', action='remove'),
    Scaler(method='standard')
]

# 3. Áp dụng
for step in steps:
    manager.apply(step)

df_clean = manager.get_data()
```

### Bước 2: Huấn luyện và So sánh mô hình

Sử dụng `model_training` để tự động tìm mô hình tốt nhất.

```python
from model_training import ModelTrainer, setup_logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Khởi tạo
setup_logging()
trainer = ModelTrainer(task_type='regression')
trainer.load_data(df_clean.drop('annual_medical_cost', axis=1), df_clean['annual_medical_cost'])
trainer.split_data()

# 2. Cấu hình các model cần đua
models_config = {
    'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100]}),
    'GradientBoosting': (GradientBoostingRegressor(), {'learning_rate': [0.01, 0.1]})
}

# 3. Chạy tự động
trainer.auto_train(models_config, output_dir='results',scaler_type='standard')
```

### Bước 3: Đánh giá và Trực quan hóa

Sử dụng `DataVisualizer` để xem kết quả.

```python
import pandas as pd
from visualize import DataVisualizer

# 1. Đọc kết quả so sánh
df_results = pd.read_csv('results/model_comparison.csv')

# 2. Vẽ biểu đồ so sánh RMSE
viz = DataVisualizer(df_results)
viz.plot_bar(x='score', y='model', title='So sánh RMSE (Thấp hơn là tốt)')
```

-----

-----


## 🖥️ Hướng dẫn cài đặt & Chạy trên Local (Máy cá nhân)

Để đảm bảo dự án chạy ổn định và không ảnh hưởng đến các dự án Python khác trong máy, chúng tôi khuyến nghị sử dụng **Môi trường ảo (Virtual Environment)**.

### Bước 1: Clone dự án về máy
Mở Terminal (hoặc CMD/PowerShell) và chạy lệnh:

```bash
# Clone repository (nếu bạn dùng git)
git clone https://github.com/nguyenhuuphuc11052005/project_python.git
cd project_python

# Hoặc nếu bạn tải file zip, hãy giải nén và mở terminal tại thư mục đó.
````

### Bước 2: Tạo môi trường ảo (Virtual Environment)

Việc này giúp cô lập các thư viện của dự án.

```bash
# Tạo môi trường ảo tên là 'venv'
python -m venv venv
```

### Bước 3: Kích hoạt môi trường ảo

Tùy thuộc vào hệ điều hành, lệnh kích hoạt sẽ khác nhau:

  * **Trên Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
  * **Trên macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

*(Sau khi kích hoạt, bạn sẽ thấy chữ `(venv)` xuất hiện ở đầu dòng lệnh)*

### Bước 4: Cài đặt các thư viện phụ thuộc

Chạy lệnh sau để cài đặt toàn bộ thư viện cần thiết từ file `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 5: Cấu trúc thư mục dữ liệu

Đảm bảo bạn đã tải file dữ liệu và đặt đúng vị trí (vì code mặc định đọc từ thư mục `data/` hoặc thư mục gốc tùy cấu hình):

```text
project_python/
│── medical_insurance.csv  <-- File dữ liệu của bạn đặt ở đây
│── preprocessing
│── model_training
│── ...
```

### Bước 6: Chạy dự án

Bạn có 2 cách để chạy:

**Cách 1: Chạy từng module (Khuyên dùng để test)**
Mỗi module đều có sẵn phần `if __name__ == "__main__":` để chạy demo.

```bash
# 1. Chạy thử quy trình xử lý dữ liệu
python demo_preprocess.py

# 2. Chạy thử quy trình huấn luyện và so sánh model
python demo_training.py

# 3. Chạy thử vẽ biểu đồ demo
python visualize.py
```

**Cách 2: Chạy trên Jupyter Notebook**
Nếu bạn muốn chạy file `EDA.ipynb`, `FE_MODELING.ipynb` để phân tích từng bước:

```bash
# Cài đặt jupyter nếu chưa có
pip install jupyterlab

# Khởi động notebook
jupyter lab
```

Sau đó mở file `EDA.ipynb` chỉnh sửa lại dòng 
```bash
%cd path_to_your_project
```

chạy (Run All) để xem EDA và xử lý missing data. Rồi sau đó mới mở file `FE_MODELING.ipynb` chỉnh sửa lại dòng
```bash
%cd path_to_your_project
```
và chạy (Run All).

------



## 📊 Kết quả thực nghiệm & So sánh (Model Evaluation)

Hệ thống đã tự động huấn luyện và so sánh nhiều thuật toán khác nhau (Linear, Tree-based, Boosting). Dưới đây là kết quả đánh giá trên tập kiểm thử (Test Set):

### 1. Bảng xếp hạng hiệu suất
*Đơn vị đo lường chính: RMSE (Root Mean Squared Error) - Càng thấp càng tốt.*


| Xếp hạng | Mô hình (Model) | RMSE | R² Score | Nhận xét chi tiết |
| :---: | :--- | :---: | :---: | :--- |
| 🏆 **1** | **XGBoost** | **0.1624** | **0.9636** | **Quán quân.** Đạt độ lỗi thấp nhất. Khả năng tối ưu hóa gradient boosting cực tốt giúp mô hình nắm bắt chính xác các mẫu dữ liệu phức tạp. |
| 🥈 2 | LightGBM | 0.1625 | 0.9635 | **Á quân.** Hiệu năng gần như ngang ngửa XGBoost (chênh lệch không đáng kể), nhưng thường có lợi thế về tốc độ huấn luyện nhanh hơn. |
| 🥉 3 | Random Forest | 0.1638 | 0.9630 | Rất ổn định. Tuy nhiên ở dataset này, phương pháp Boosting (XGBoost/LightGBM) đã chứng minh hiệu quả hơn phương pháp Bagging. |
| 4 | Gradient Boosting | 0.1651 | 0.9624 | Hiệu quả cao, xếp ngay sau top 3. Là nền tảng tốt nhưng chưa tối ưu bằng các phiên bản cải tiến như XGB/LGBM. |
| 5 | Decision Tree | 0.1706 | 0.9598 | Khá ấn tượng đối với một mô hình đơn lẻ, nhưng vẫn thua kém các mô hình tổ hợp (Ensemble) do khả năng tổng quát hóa kém hơn. |
| 6 | AdaBoost | 0.2050 | 0.9420 | Hiệu suất trung bình khá. Cơ chế trọng số thích nghi chưa phát huy tác dụng tối đa so với Gradient Boosting ở bài toán này. |
| 7 | Ridge Regression | 0.2180 | 0.9344 | Tốt hơn Linear Regression một chút xíu nhờ Regularization, nhưng vẫn không bắt được các mối quan hệ phi tuyến tính. |
| 8 | Linear Regression | 0.2180 | 0.9344 | Mô hình cơ sở (Baseline). Hiệu suất thấp hơn nhóm cây quyết định, cho thấy dữ liệu có tính phi tuyến cao. |
| 9 | Lasso Regression | 0.2884 | 0.8852 | **Kém nhất.** Việc triệt tiêu các biến (Feature Selection mạnh tay) dường như đã làm mất đi nhiều thông tin quan trọng, dẫn đến underfitting. |

*(Lưu ý: RMSE được tính trên biến mục tiêu `annual_medical_cost` đã qua xử lý Log-transform)*



### 3. Phân tích kết quả
* **Chiến thắng của Tree-based Models:** Random Forest , LightGBM, XGBoost vượt trội vì dữ liệu y tế chứa nhiều ngưỡng (thresholds) và tương tác phi tuyến. Ví dụ: BMI chỉ thực sự làm tăng vọt chi phí khi vượt qua mức 30 (béo phì) và đi kèm với việc hút thuốc. Linear Regression khó học được điều này nếu không tạo biến tương tác thủ công.
* **Độ ổn định:** Random Forest cho thấy độ biến thiên thấp (Low Variance) khi kiểm thử chéo (Cross-validation), chứng tỏ mô hình ít bị Overfitting.



| Metric | Giá trị (Log Scale) | Ý nghĩa |
| :--- | :--- | :--- |
| **RMSE** | \~0.1624 | Sai số trung bình phương căn (Root Mean Squared Error) |
| **MAE** | \~0.1291 | Sai số tuyệt đối trung bình |
| **R²** | \~0.9636 | Mức độ giải thích độ biến thiên dữ liệu |
| **MAPE**| \~0.0171 | Sai số phần trăm trung bình|

-----


## 📝 Ghi chú cho Google Colab

Nếu chạy trên Google Colab, hãy upload 3 file module (`preprocess.py`, `visualize.py`, `model_trainer.py`) vào cùng thư mục với Notebook, hoặc mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
import sys
%cd /content/drive/MyDrive/path_to_your_project
```

## 🚀 Hướng phát triển tiếp theo (Roadmap)

Dù mô hình hiện tại đã đạt kết quả tốt (RMSE ~0.16), dự án vẫn có thể cải thiện thêm:

* **Deploy Model:** Xây dựng API bằng **FastAPI** hoặc **Flask** để phục vụ dự đoán realtime.
* **Dockerize:** Đóng gói toàn bộ môi trường chạy vào Docker Container để dễ dàng triển khai.
* **Feature Selection nâng cao:** Sử dụng SHAP values để giải thích mô hình rõ ràng hơn (Explainable AI).
* **Deep Learning:** Thử nghiệm mạng nơ-ron (Neural Network) với Keras/TensorFlow để xem có vượt qua được Random Forest không.


## 👥 Tác giả

  * **Họ và tên:** Nguyễn Hữu Phước, MSSV: 23280078
  * **Họ và tên:** Nguyễn Chí Tiến, MSSV: 23280087
 

-----