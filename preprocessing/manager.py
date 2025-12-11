import pandas as pd
from typing import Union, Optional
from .base import BasePreprocessor

class DataManager:
    """Lớp quản lý dữ liệu."""
    
    def __init__(self, input_data: Union[str, pd.DataFrame, None] = None) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.filepath: Optional[str] = None

        if input_data is not None:
            if isinstance(input_data, str):
                self.filepath = input_data
                self.load_data(input_data)
            elif isinstance(input_data, pd.DataFrame):
                self.df = input_data.copy()
                # print(f"--> [DataManager] Đã khởi tạo từ DataFrame có sẵn. Shape: {self.df.shape}")
            else:
                print("Lỗi: Đầu vào phải là đường dẫn file (str) hoặc DataFrame.")

    def load_data(self, filepath: str) -> None:
        try:
            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                self.df = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                self.df = pd.read_json(filepath)
            else:
                raise ValueError("Định dạng file không hỗ trợ.")
            # print(f"--> [DataManager] Đã tải dữ liệu từ file. Shape: {self.df.shape}")
        except Exception as e:
            print(f"Lỗi đọc file '{filepath}': {e}")
            self.df = pd.DataFrame()

    def apply(self, preprocessor: BasePreprocessor) -> None:
        if self.df is None or self.df.empty:
            print("Cảnh báo: Không có dữ liệu để xử lý.")
            return

        print(f"\n--- Đang áp dụng: {preprocessor.__class__.__name__} ---")
        self.df = preprocessor.process(self.df)
        print(f"-> Số lượng NaN còn lại: {preprocessor.missing_count}")

    def get_data(self) -> pd.DataFrame:
        return self.df
    
    def save_data(self, output_path: str) -> None:
        if self.df is None or self.df.empty:
            print("Lỗi: Không có dữ liệu để lưu.")
            return
            
        try:
            if output_path.endswith('.csv'):
                self.df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                self.df.to_excel(output_path, index=False)
            elif output_path.endswith('.json'):
                self.df.to_json(output_path, orient='records')
            else:
                print("Lỗi: Định dạng file không hỗ trợ (chỉ csv, xlsx, json).")
                return
            print(f"--> [DataManager] Đã lưu dữ liệu vào '{output_path}'.")
        except Exception as e:
            print(f"Lỗi khi lưu file: {e}")