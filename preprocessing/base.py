import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class BasePreprocessor(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các bộ xử lý dữ liệu."""

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._last_missing_count: int = 0

    def __repr__(self) -> str:
        status = "Fitted" if self._is_fitted else "Unfitted"
        return f"<{self.__class__.__name__} (Status: {status})>"

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def missing_count(self) -> int:
        return self._last_missing_count

    def _update_state(self, df: pd.DataFrame) -> None:
        self._is_fitted = True
        self._last_missing_count = int(df.isna().sum().sum())

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric, categorical