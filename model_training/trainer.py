import pandas as pd
import numpy as np
import logging
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, mean_squared_error, mean_absolute_error, 
                             r2_score, mean_absolute_percentage_error)
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class ModelTrainer:
    """
    Lớp quản lý quy trình huấn luyện, tối ưu và đánh giá mô hình học máy.
    Sử dụng Scikit-learn Pipeline để đóng gói quy trình Scaling và Modeling.
    """

    def __init__(self, random_state: int = 42, task_type: str = 'classification'):
        self.random_state = random_state
        self.task_type = task_type
        self.model = None # Lúc này model sẽ là một Pipeline object
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

    # --- HÀM scale_data CŨ ĐÃ BỊ LOẠI BỎ ---

    def optimize_params(self, estimator: BaseEstimator, param_grid: dict, method: str = 'grid', cv: int = 3, scaler_type: str = 'standard'):
        """
        Tối ưu siêu tham số sử dụng Pipeline.
        Pipeline gồm 2 bước: Scaler -> Model.
        """
        logging.info(f"Bắt đầu tối ưu tham số cho {estimator.__class__.__name__} ({method}) với {scaler_type} scaler...")
        
        # 1. Chọn Scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = None # Không scale

        # 2. Tạo Pipeline
        steps = []
        if scaler:
            steps.append(('scaler', scaler))
        steps.append(('model', estimator))
        
        pipeline = Pipeline(steps)

        # 3. Điều chỉnh param_grid để khớp với tên trong pipeline
        # Ví dụ: 'n_estimators' -> 'model__n_estimators'
        pipeline_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
        
        scoring = 'accuracy' if self.task_type == 'classification' else 'neg_root_mean_squared_error'

        # 4. Chạy Search trên Pipeline
        if method == 'grid':
            search = GridSearchCV(pipeline, pipeline_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        elif method == 'random':
            search = RandomizedSearchCV(pipeline, pipeline_param_grid, n_iter=10, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, random_state=self.random_state)
        
        search.fit(self.X_train, self.y_train)
        
        self.model = search.best_estimator_ # Lưu lại pipeline tốt nhất
        self.best_params = search.best_params_
        logging.info(f"Tối ưu hoàn tất. Best params: {self.best_params}")
        return self.model

    def evaluate(self, output_dir: str = 'results'):
        """Đánh giá mô hình (Pipeline)."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Pipeline sẽ tự động scale X_test trước khi predict
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
            mae = mean_absolute_error(self.y_test, y_pred)
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            self.metrics = {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}
            logging.info(f"Kết quả đánh giá: RMSE={rmse:.4f}, R2={r2:.4f}")

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
        """Lưu Pipeline (đã bao gồm model và scaler)."""
        if self.model is None:
            logging.error("Không có mô hình để lưu.")
            return
        
        artifacts = {
            'pipeline': self.model, # Lưu object pipeline
            'metrics': self.metrics,
            'best_params': self.best_params
        }
        
        joblib.dump(artifacts, filepath)
        logging.info(f"Đã lưu trọn gói Pipeline tại: {filepath}")

    def load_model_from_file(self, filepath: str):
        """Load Pipeline từ file."""
        try:
            artifacts = joblib.load(filepath)
            
            if isinstance(artifacts, dict) and 'pipeline' in artifacts:
                self.model = artifacts['pipeline']
                self.metrics = artifacts.get('metrics', {})
                self.best_params = artifacts.get('best_params', {})
                logging.info(f"Đã load Pipeline từ: {filepath}")
            else:
                # Hỗ trợ format cũ nếu cần
                self.model = artifacts.get('model', artifacts)
                logging.warning("Đã load Model (có thể là format cũ không phải pipeline).")
                
        except FileNotFoundError:
            logging.error(f"Không tìm thấy file: {filepath}")

    def auto_train(self, models_dict: dict, output_dir: str = 'results', scaler_type: str = 'standard'):
        """Tự động huấn luyện và so sánh nhiều mô hình dùng Pipeline."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        best_overall_score = -float('inf') if self.task_type == 'classification' else float('inf')
        best_model_name = ""
        results = []

        for name, (model, params) in models_dict.items():
            logging.info(f"--- Auto Train: {name} ---")
            
            # Truyền thêm loại scaler vào hàm optimize
            self.optimize_params(model, params, method='random', cv=3, scaler_type=scaler_type)
            
            # Đánh giá trên tập test (Pipeline tự lo việc scale)
            y_pred = self.model.predict(self.X_test)
            
            if self.task_type == 'classification':
                score = accuracy_score(self.y_test, y_pred)
                is_better = score > best_overall_score
                current_metrics = {'accuracy': score} 
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                mape = mean_absolute_percentage_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                current_metrics = {'mape': mape, 'rmse': rmse, 'mae': mae, 'r2': r2}
                score = rmse 
                is_better = score < best_overall_score

            # Lưu kết quả
            results.append({
                  'model': name, 
                  'score': score, 
                  'best_params': self.best_params,
                  **current_metrics 
            })

            logging.info(f"{name} - Score: {score:.4f}")

            if is_better:
                best_overall_score = score
                best_model_name = name
                self.metrics = current_metrics
                # Lưu pipeline tốt nhất
                self.save_model(os.path.join(output_dir, 'best_model.pkl'))

        # Lưu CSV so sánh
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        logging.info(f"BEST MODEL: {best_model_name} (Score: {best_overall_score:.4f})")
        
        # Load lại model tốt nhất để sẵn sàng sử dụng
        self.load_model_from_file(os.path.join(output_dir, 'best_model.pkl'))