import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base import BasePreprocessor

class Scaler(BasePreprocessor):
    """Lớp chuyên trách chuẩn hóa dữ liệu số."""

    def __init__(self, method: str = 'standard') -> None:
        super().__init__()
        self.method = method
        self.scaler = StandardScaler() if method == 'standard' else MinMaxScaler()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        numeric_cols, _ = self.get_column_types(df_processed)
        
        if not numeric_cols:
            print("[Scaler] Cảnh báo: Không có cột số nào để chuẩn hóa.")
            return df_processed

        try:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
            print(f"[Scaler] Đã chuẩn hóa {len(numeric_cols)} cột bằng phương pháp '{self.method}'.")
            self._update_state(df_processed)
            return df_processed
        except Exception as e:
            print(f"Lỗi trong Scaler: {e}")
            return df