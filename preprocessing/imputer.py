import pandas as pd
from typing import List, Optional, Any
from .base import BasePreprocessor

class Imputer(BasePreprocessor):
    """Lớp xử lý giá trị thiếu."""
    
    def __init__(self, 
                 strategy: str = 'mean', 
                 fill_value: Optional[Any] = None, 
                 columns: Optional[List[str]] = None) -> None:
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        if self.columns:
            target_cols = [c for c in self.columns if c in df_processed.columns]
        else:
            numeric_cols, _ = self.get_column_types(df_processed)
            if self.strategy in ['mean', 'median']:
                target_cols = numeric_cols
            else:
                target_cols = df_processed.columns.tolist()

        try:
            if self.strategy == 'constant':
                if self.fill_value is None:
                    raise ValueError("Cần fill_value cho strategy='constant'")
                df_processed[target_cols] = df_processed[target_cols].fillna(self.fill_value)
                
            elif self.strategy == 'mode':
                for col in target_cols:
                    if not df_processed[col].mode().empty:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
            elif self.strategy == 'mean':
                for col in target_cols:
                    if not df_processed[col].mean().empty:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean()[0])

            elif self.strategy == 'median':
                for col in target_cols:
                    if not df_processed[col].median().empty:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median()[0])
            
            elif self.strategy == 'ffill':
                df_processed[target_cols] = df_processed[target_cols].ffill()
            
            print(f"[Imputer] Đã điền '{self.fill_value if self.strategy=='constant' else self.strategy}' vào các cột: {target_cols}")
            self._update_state(df_processed)
            return df_processed

        except Exception as e:
            print(f"Lỗi trong Imputer: {e}")
            return df