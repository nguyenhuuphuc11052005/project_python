import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.ensemble import IsolationForest
from .base import BasePreprocessor

class OutlierHandler(BasePreprocessor):
    """Lớp chuyên trách phát hiện và xử lý dữ liệu ngoại lai."""

    def __init__(self, 
                 method: str = 'iqr', 
                 action: str = 'remove', 
                 threshold: float = 1.5, 
                 columns: Optional[List[str]] = None) -> None:
        super().__init__()
        self.method = method
        self.action = action
        self.threshold = threshold
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        numeric_cols, _ = self.get_column_types(df_processed)
        
        if self.columns:
            target_cols = [c for c in self.columns if c in numeric_cols]
            invalid_cols = set(self.columns) - set(target_cols)
            if invalid_cols:
                print(f"[OutlierHandler] Cảnh báo: Các cột sau không phải số: {invalid_cols}")
        else:
            target_cols = numeric_cols

        if not target_cols:
            print("[OutlierHandler] Không có cột số nào để xử lý.")
            return df_processed

        outlier_indices = set()

        try:
            if self.method == 'isolation_forest':
                temp_data = df_processed[target_cols].fillna(df_processed[target_cols].median())
                clf = IsolationForest(contamination=0.05, random_state=42)
                preds = clf.fit_predict(temp_data)
                outlier_rows = df_processed.index[preds == -1]
                
                if self.action == 'remove':
                    outlier_indices.update(outlier_rows)
                elif self.action == 'set_nan':
                    df_processed.loc[outlier_rows, target_cols] = np.nan
                print(f"[OutlierHandler] Isolation Forest tìm thấy {len(outlier_rows)} dòng ngoại lai.")
            
            else:
                for col in target_cols:
                    lower, upper = self._calculate_bounds(df_processed[col])
                    
                    if self.action == 'remove':
                        outliers = df_processed[(df_processed[col] < lower) | (df_processed[col] > upper)].index
                        outlier_indices.update(outliers)
                    elif self.action == 'cap':
                        df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
                    elif self.action == 'set_nan':
                        mask = (df_processed[col] < lower) | (df_processed[col] > upper)
                        df_processed.loc[mask, col] = np.nan

            if self.action == 'remove' and outlier_indices:
                df_processed = df_processed.drop(index=list(outlier_indices))
                print(f"[OutlierHandler] Đã xóa {len(outlier_indices)} dòng chứa ngoại lai.")
            elif self.action == 'cap':
                print(f"[OutlierHandler] Đã thay thế ngoại lai bằng biên (Capping) trên các cột: {target_cols}")
            elif self.action == 'set_nan':
                print(f"[OutlierHandler] Đã thay thế ngoại lai bằng NaN trên các cột: {target_cols}")

            self._update_state(df_processed)
            return df_processed

        except Exception as e:
            print(f"Lỗi trong OutlierHandler: {e}")
            return df

    def _calculate_bounds(self, series: pd.Series) -> tuple[float, float]:
        if self.method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return Q1 - self.threshold * IQR, Q3 + self.threshold * IQR
        elif self.method == 'zscore':
            mean = series.mean()
            std = series.std()
            return mean - self.threshold * std, mean + self.threshold * std
        else:
            raise ValueError(f"Phương pháp '{self.method}' không hợp lệ.")