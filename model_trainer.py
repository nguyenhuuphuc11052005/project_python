import pandas as pd
import numpy as np
import logging
import joblib
import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Import các mô hình và metrics từ sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, roc_curve, auc, mean_squared_error)
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout) # Thêm sys.stdout để chắc chắn in ra màn hình Colab
    ],
    force=True  # <--- QUAN TRỌNG: Thêm dòng này để ghi đè cấu hình của Colab
)

class ModelTrainer:
    """
    Lớp quản lý quy trình huấn luyện, tối ưu và đánh giá mô hình học máy.
    Hỗ trợ Classification (mặc định) và Regression cơ bản.
    Đã tích hợp Scaling.
    """

    def __init__(self, random_state: int = 42, task_type: str = 'classification'):
        self.random_state = random_state
        self.task_type = task_type
        self.model = None
        self.scaler = None  # Thuộc tính mới để lưu scaler
        self.best_params = {}
        self.metrics = {}
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, X: pd.DataFrame, y: pd.Series):
        """Nạp dữ liệu features và target."""
        self.X = X
        self.y = y
        logging.info(f"Đã nạp dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng.")

    def split_data(self, test_size: float = 0.2):
        """Chia dữ liệu train/test."""
        if self.X is None or self.y is None:
            raise ValueError("Dữ liệu chưa được nạp. Hãy gọi load_data() trước.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, 
            stratify=self.y if self.task_type == 'classification' else None
        )
        logging.info(f"Chia dữ liệu: Train={self.X_train.shape}, Test={self.X_test.shape}")

    def scale_data(self, method: str = 'standard'):
        """
        Chuẩn hóa dữ liệu (Scaling).
        QUAN TRỌNG: Chỉ fit trên tập Train, sau đó transform tập Test để tránh Data Leakage.
        
        Args:
            method (str): 'standard', 'minmax', hoặc 'robust'.
        """
        if self.X_train is None:
            raise ValueError("Chưa chia dữ liệu. Vui lòng gọi split_data() trước.")

        logging.info(f"Đang thực hiện scaling dữ liệu theo phương pháp: {method}")
        
        # Chọn Scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            logging.warning(f"Phương pháp {method} không hỗ trợ. Bỏ qua scaling.")
            return

        # Fit và Transform trên tập Train
        # Chuyển đổi về DataFrame để giữ tên cột (tùy chọn, giúp debug dễ hơn)
        cols = self.X_train.columns
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=cols)
        
        # Chỉ Transform trên tập Test (Dùng tham số đã học từ Train)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=cols)
        
        logging.info("Scaling hoàn tất.")

    def optimize_params(self, estimator: BaseEstimator, param_grid: dict, method: str = 'grid', cv: int = 5):
        """Tối ưu siêu tham số."""
        logging.info(f"Bắt đầu tối ưu tham số cho {estimator.__class__.__name__} ({method})...")
        
        if method == 'grid':
            search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        elif method == 'random':
            search = RandomizedSearchCV(estimator, param_grid, n_iter=10, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1, random_state=self.random_state)
        
        search.fit(self.X_train, self.y_train)
        
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        logging.info(f"Tối ưu hoàn tất. Best params: {self.best_params}")
        return self.model

    def evaluate(self, output_dir: str = 'results'):
        """Đánh giá mô hình."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        y_pred = self.model.predict(self.X_test)
        
        if self.task_type == 'classification':
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            report = classification_report(self.y_test, y_pred, output_dict=True)
            self.metrics = {'accuracy': acc, 'f1_score': f1, 'detail': report}
            logging.info(f"Kết quả đánh giá: Accuracy={acc:.4f}, F1={f1:.4f}")
            self._plot_confusion_matrix(self.y_test, y_pred, output_dir)
        elif self.task_type == 'regression':
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            self.metrics = {'mse': mse, 'rmse': rmse}
            logging.info(f"Kết quả đánh giá: RMSE={rmse:.4f}")

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def _plot_confusion_matrix(self, y_true, y_pred, output_dir):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    def save_model(self, filepath: str):
        """
        Lưu cả Model và Scaler vào một file dictionary.
        Điều này đảm bảo khi load lại, ta có thể tiền xử lý dữ liệu mới chính xác như lúc train.
        """
        if self.model is None:
            logging.error("Không có mô hình để lưu.")
            return
        
        # Đóng gói artifacts
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,  # Lưu kèm scaler
            'metrics': self.metrics,
            'best_params': self.best_params
        }
        
        joblib.dump(artifacts, filepath)
        logging.info(f"Đã lưu trọn gói Model + Scaler tại: {filepath}")

    def load_model_from_file(self, filepath: str):
        """
        Load model và scaler từ file, cập nhật vào instance hiện tại.
        """
        try:
            artifacts = joblib.load(filepath)
            
            # Kiểm tra xem file có phải format mới (dict) hay cũ (model object)
            if isinstance(artifacts, dict) and 'model' in artifacts:
                self.model = artifacts['model']
                self.scaler = artifacts.get('scaler')
                self.metrics = artifacts.get('metrics', {})
                logging.info(f"Đã load Model và Scaler từ: {filepath}")
                if self.scaler:
                    logging.info(f"Scaler type: {type(self.scaler).__name__}")
                else:
                    logging.warning("File này không chứa Scaler!")
            else:
                # Fallback cho format cũ (chỉ lưu model object)
                self.model = artifacts
                self.scaler = None
                logging.warning("Đã load Model (định dạng cũ, không có Scaler/Metrics).")
                
        except FileNotFoundError:
            logging.error(f"Không tìm thấy file: {filepath}")

    def auto_train(self, models_dict: dict, output_dir: str = 'results'):
        best_overall_score = -float('inf') if self.task_type == 'classification' else float('inf')
        best_model_name = ""
        results = []

        for name, (model, params) in models_dict.items():
            logging.info(f"--- Auto Train: {name} ---")
            self.optimize_params(model, params, method='random', cv=3)
            
            y_pred = self.model.predict(self.X_test)
            
            if self.task_type == 'classification':
                score = accuracy_score(self.y_test, y_pred)
                is_better = score > best_overall_score
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                mape = mean_absolute_percentage_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                current_metrics = {'mape': mape, 'rmse': rmse, 'mae': mae, 'r2': r2}
                score = rmse 
                is_better = score < best_overall_score

                # Lưu tất cả chỉ số vào dictionary
                results.append({
                  'model': name, 
                  'score': score, 
                  'best_params': self.best_params,
                  **current_metrics 
            })

            logging.info(f"{name} - RMSE: {score:.4f} | R2: {r2:.4f}")

            results.append({'model': name, 'score': score, 'best_params': self.best_params})
         

            if is_better:
                best_overall_score = score
                best_model_name = name

                self.metrics = current_metrics
                
                self.save_model(os.path.join(output_dir, 'best_model.pkl'))

        # Lưu CSV với đầy đủ cột mới
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        logging.info(f"BEST MODEL: {best_model_name} (RMSE: {best_overall_score:.4f})")
        
        self.load_model_from_file(os.path.join(output_dir, 'best_model.pkl'))


# --- Phần chạy script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'auto'], default='train')
    parser.add_argument('--scaler', type=str, choices=['standard', 'minmax', 'robust', 'none'], default='standard', help='Loại scaler')
    args = parser.parse_args()

    # Tạo dữ liệu giả
    from sklearn.datasets import make_classification
    # Tạo dữ liệu có sự chênh lệch lớn về scale để test hiệu quả của Scaler
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_dummy[:, 0] = X_dummy[:, 0] * 1000  # Nhân feature 0 lên để tạo scale lớn
    X_dummy[:, 1] = X_dummy[:, 1] * 0.001 # Feature 1 rất nhỏ
    
    df = pd.DataFrame(X_dummy, columns=[f'feat_{i}' for i in range(10)])
    df['target'] = y_dummy
    
    trainer = ModelTrainer(random_state=42)
    
    # 1. Load & Split
    trainer.load_data(df.drop(columns=['target']), df['target'])
    trainer.split_data()
    
    # 2. Scale Data (Bước mới thêm vào)
    if args.scaler != 'none':
        trainer.scale_data(method=args.scaler)
        # In thử để kiểm tra scale
        print("Mean của X_train sau khi scale (mong đợi gần 0 nếu là Standard):")
        print(trainer.X_train.mean().head(2)) 

    # 3. Train
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    if args.mode == 'train':
        # SVM cực kỳ nhạy cảm với Scale, nếu không scale acc sẽ thấp
        svm = SVC()
        params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        trainer.optimize_params(svm, params, method='random')
        trainer.evaluate()
        trainer.save_model("final_model.pkl")

    elif args.mode == 'auto':
        models = {
            'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100]}),
            'SVM': (SVC(), {'C': [1, 10]}) # SVM sẽ hưởng lợi lớn từ việc scale
        }
        trainer.auto_train(models)
        trainer.evaluate()