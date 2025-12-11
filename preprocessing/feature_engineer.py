import pandas as pd
from typing import List, Optional, Dict
from sklearn.preprocessing import OrdinalEncoder
from .base import BasePreprocessor

class FeatureEngineer(BasePreprocessor):
    """Lớp chuyên trách tạo đặc trưng mới và mã hóa dữ liệu."""

    def __init__(self, 
                 datetime_col: Optional[str] = None, 
                 text_mapping: Optional[Dict[str, Dict[str, int]]] = None,
                 ordinal_cols: Optional[Dict[str, List[str]]] = None,
                 one_hot_cols: Optional[List[str]] = None) -> None:
        super().__init__()
        self.datetime_col = datetime_col
        self.text_mapping = text_mapping
        self.ordinal_cols = ordinal_cols
        self.one_hot_cols = one_hot_cols
        self._ordinal_encoders = {} 

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        try:
            if self.datetime_col and self.datetime_col in df_processed.columns:
                col = self.datetime_col
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                df_processed[f'{col}_year'] = df_processed[col].dt.year
                df_processed[f'{col}_month'] = df_processed[col].dt.month
                df_processed[f'{col}_day'] = df_processed[col].dt.day
                df_processed.drop(columns=[col], inplace=True)
                print(f"[FeatureEngineer] Đã tách ngày tháng cho cột '{col}'.")

            if self.text_mapping:
                for col, mapping in self.text_mapping.items():
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].map(mapping).fillna(df_processed[col])
                        print(f"[FeatureEngineer] Đã map giá trị thủ công cho cột '{col}'.")

            if self.ordinal_cols:
                df_processed = self._apply_ordinal_encoding(df_processed)

            if self.one_hot_cols:
                df_processed = self._apply_one_hot_encoding(df_processed)

            self._update_state(df_processed)
            return df_processed
        except Exception as e:
            print(f"Lỗi trong FeatureEngineer: {e}")
            return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, categories in self.ordinal_cols.items():
            if col in df.columns:
                encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
                encoded_data = encoder.fit_transform(df[[col]])
                df[col] = encoded_data.flatten()
                self._ordinal_encoders[col] = encoder
                print(f"[FeatureEngineer] Đã Ordinal Encode cột '{col}' với thứ tự: {categories}")
            else:
                print(f"[FeatureEngineer] Cảnh báo: Cột '{col}' không tìm thấy để Ordinal Encode.")
        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_cols = [c for c in self.one_hot_cols if c in df.columns]
        if valid_cols:
            df = pd.get_dummies(df, columns=valid_cols, dtype=int)
            print(f"[FeatureEngineer] Đã One-Hot Encode các cột: {valid_cols}")
        else:
            print("[FeatureEngineer] Không tìm thấy cột nào để One-Hot Encode.")
        return df