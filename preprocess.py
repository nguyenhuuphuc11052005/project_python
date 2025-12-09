import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Optional, Dict, Union, Any
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest

# =============================================================================
# 1. BASE CLASS (Lớp Trừu Tượng)
# =============================================================================

class BasePreprocessor(ABC):
    """
    Lớp cơ sở trừu tượng (Abstract Base Class) cho tất cả các bộ xử lý dữ liệu.
    
    Lớp này định nghĩa giao diện chung (interface) và các phương thức tiện ích
    mà các lớp con (Imputer, Scaler, v.v.) cần tuân theo.
    """

    def __init__(self) -> None:
        """Khởi tạo các thuộc tính trạng thái cơ bản."""
        self._is_fitted: bool = False
        self._last_missing_count: int = 0

    def __repr__(self) -> str:
        """Trả về chuỗi đại diện cho đối tượng."""
        status = "Fitted" if self._is_fitted else "Unfitted"
        return f"<{self.__class__.__name__} (Status: {status})>"

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phương thức trừu tượng để xử lý dữ liệu.
        
        Args:
            df (pd.DataFrame): DataFrame đầu vào cần xử lý.
            
        Returns:
            pd.DataFrame: DataFrame sau khi đã được xử lý.
        """
        pass

    @property
    def missing_count(self) -> int:
        """
        Thuộc tính tính toán (Computed Property).
        
        Returns:
            int: Tổng số lượng giá trị thiếu (NaN) trong DataFrame 
                 tại lần xử lý gần nhất.
        """
        return self._last_missing_count

    def _update_state(self, df: pd.DataFrame) -> None:
        """
        Cập nhật trạng thái nội bộ sau khi xử lý xong (Protected method).
        
        Args:
            df (pd.DataFrame): DataFrame sau khi xử lý.
        """
        self._is_fitted = True
        self._last_missing_count = int(df.isna().sum().sum())

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Phân loại các cột trong DataFrame thành nhóm số và nhóm phân loại.
        
        Args:
            df (pd.DataFrame): DataFrame cần phân tích.
            
        Returns:
            Tuple[List[str], List[str]]: 
                - Danh sách tên các cột số (numeric).
                - Danh sách tên các cột phân loại (categorical/object).
        """
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric, categorical


# =============================================================================
# 2. CONCRETE CLASSES (Các Lớp Con Cụ Thể)
# =============================================================================

class Imputer(BasePreprocessor):
    """
    Lớp xử lý giá trị thiếu, hỗ trợ chọn cột cụ thể.
    """
    
    def __init__(self, 
                 strategy: str = 'mean', 
                 fill_value: Optional[Any] = None, 
                 columns: Optional[List[str]] = None) -> None:
        """
        Args:
            strategy: 'mean', 'median', 'mode', 'constant', 'drop'.
            fill_value: Giá trị điền khi strategy='constant'.
            columns: Danh sách cột cần xử lý. Nếu None, tự động chọn tất cả phù hợp.
        """
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        # 1. Xác định cột mục tiêu
        if self.columns:
            # Nếu người dùng chỉ định cột, chỉ lấy các cột có trong DF
            target_cols = [c for c in self.columns if c in df_processed.columns]
        else:
            # Logic tự động cũ
            numeric_cols, _ = self.get_column_types(df_processed)
            if self.strategy in ['mean', 'median']:
                target_cols = numeric_cols
            else:
                target_cols = df_processed.columns.tolist()

        try:
            # 2. Xử lý
            if self.strategy == 'constant':
                if self.fill_value is None:
                    raise ValueError("Cần fill_value cho strategy='constant'")
                
                # Chỉ điền cho các cột target
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
                # Forward fill: điền giá trị phía trước vào chỗ trống
                df_processed[target_cols] = df_processed[target_cols].ffill()
            print(f"[Imputer] Đã điền '{self.fill_value if self.strategy=='constant' else self.strategy}' vào các cột: {target_cols}")
            self._update_state(df_processed)
            return df_processed

        except Exception as e:
            print(f"Lỗi trong Imputer: {e}")
            return df


class Scaler(BasePreprocessor):
    """
    Lớp chuyên trách chuẩn hóa dữ liệu số (Normalization/Standardization).
    """

    def __init__(self, method: str = 'standard') -> None:
        """
        Khởi tạo bộ chuẩn hóa.

        Args:
            method (str): Phương pháp chuẩn hóa.
                'standard': Sử dụng StandardScaler (z-score).
                'minmax': Sử dụng MinMaxScaler (về khoảng [0, 1]).
                Mặc định là 'standard'.
        """
        super().__init__()
        self.method = method
        self.scaler = StandardScaler() if method == 'standard' else MinMaxScaler()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thực hiện chuẩn hóa các cột số."""
        df_processed = df.copy()
        numeric_cols, _ = self.get_column_types(df_processed)
        
        if not numeric_cols:
            print("[Scaler] Cảnh báo: Không có cột số nào để chuẩn hóa.")
            return df_processed

        try:
            # Fit và Transform trên các cột số
            # Lưu ý: scaler của sklearn sẽ trả về numpy array, cần gán lại vào DF
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
            
            print(f"[Scaler] Đã chuẩn hóa {len(numeric_cols)} cột bằng phương pháp '{self.method}'.")
            self._update_state(df_processed)
            return df_processed
        except Exception as e:
            print(f"Lỗi trong Scaler: {e}")
            return df


class OutlierHandler(BasePreprocessor):
    """
    Lớp chuyên trách phát hiện và xử lý dữ liệu ngoại lai (Outliers).
    Hỗ trợ xử lý trên các cột chỉ định và nhiều phương pháp xử lý khác nhau.
    """

    def __init__(self, 
                 method: str = 'iqr', 
                 action: str = 'remove', 
                 threshold: float = 1.5, 
                 columns: Optional[List[str]] = None) -> None:
        """
        Khởi tạo bộ xử lý ngoại lai.

        Args:
            method (str): Phương pháp phát hiện ('iqr', 'zscore','isolation_forest').
            action (str): Hành động xử lý ngoại lai.
                'remove': Xóa các hàng chứa ngoại lai.
                'cap': (Capping/Clipping) Thay thế ngoại lai bằng giá trị biên.
                'set_nan': Thay thế ngoại lai bằng giá trị NaN (để Imputer xử lý).
                Mặc định là 'remove'.
            threshold (float): Ngưỡng xác định (1.5 cho IQR, 3.0 cho Z-score).
            columns (Optional[List[str]]): Danh sách tên cột cụ thể cần xử lý. 
                Nếu None, sẽ tự động áp dụng cho tất cả các cột số.
        """
        super().__init__()
        self.method = method
        self.action = action
        self.threshold = threshold
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thực hiện xử lý ngoại lai."""
        df_processed = df.copy()
        
        # 1. Xác định các cột cần xử lý
        numeric_cols, _ = self.get_column_types(df_processed)
        
        # Nếu người dùng chỉ định columns, lọc lấy giao của columns đó và numeric_cols
        if self.columns:
            # Kiểm tra xem cột người dùng nhập có tồn tại và là số không
            target_cols = [c for c in self.columns if c in numeric_cols]
            invalid_cols = set(self.columns) - set(target_cols)
            if invalid_cols:
                print(f"[OutlierHandler] Cảnh báo: Các cột sau không phải số hoặc không tồn tại: {invalid_cols}")
        else:
            target_cols = numeric_cols

        if not target_cols:
            print("[OutlierHandler] Không có cột số nào để xử lý.")
            return df_processed

        original_len = len(df_processed)
        outlier_indices = set() # Dùng để lưu index các dòng cần xóa (nếu action='remove')

        try:
            if self.method == 'isolation_forest':
                # Cần fillna tạm thời để model chạy được (Isolation Forest không nhận NaN)
                temp_data = df_processed[target_cols].fillna(df_processed[target_cols].median())
                
                clf = IsolationForest(contamination=0.05, random_state=42)
                preds = clf.fit_predict(temp_data) # -1 là outlier, 1 là normal
                
                # Lấy index của outlier
                outlier_rows = df_processed.index[preds == -1]
                
                if self.action == 'remove':
                    outlier_indices.update(outlier_rows)
                elif self.action == 'set_nan':
                    df_processed.loc[outlier_rows, target_cols] = np.nan
                
                print(f"[OutlierHandler] Isolation Forest tìm thấy {len(outlier_rows)} dòng ngoại lai.")
            
            else:
                for col in target_cols:
                    # Tính toán cận trên và cận dưới (Bounds)
                    lower_bound, upper_bound = self._calculate_bounds(df_processed[col])
                    
                    # Logic xử lý dựa trên action
                    if self.action == 'remove':
                        # Tìm các index vi phạm
                        outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)].index
                        outlier_indices.update(outliers)
                    
                    elif self.action == 'cap':
                        # Thay thế giá trị < lower bằng lower, > upper bằng upper
                        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    elif self.action == 'set_nan':
                        # Thay thế giá trị ngoại lai bằng np.nan
                        mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                        df_processed.loc[mask, col] = np.nan

            # Thực hiện xóa hàng nếu action là 'remove'
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
        """
        Hàm hỗ trợ tính toán biên dưới và biên trên.
        """
        if self.method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            return lower, upper
            
        elif self.method == 'zscore':
            mean = series.mean()
            std = series.std()
            # Z = (X - mean) / std => X = Z * std + mean
            lower = mean - self.threshold * std
            upper = mean + self.threshold * std
            return lower, upper
        
        else:
            raise ValueError(f"Phương pháp '{self.method}' không hợp lệ.")




class FeatureEngineer(BasePreprocessor):
    """
    Lớp chuyên trách tạo đặc trưng mới (Feature Engineering) và mã hóa dữ liệu.
    Nâng cấp: Hỗ trợ Ordinal Encoding và One-Hot Encoding.
    """

    def __init__(self, 
                 datetime_col: Optional[str] = None, 
                 text_mapping: Optional[Dict[str, Dict[str, int]]] = None,
                 ordinal_cols: Optional[Dict[str, List[str]]] = None,
                 one_hot_cols: Optional[List[str]] = None) -> None:
        """
        Args:
            datetime_col: Tên cột thời gian cần tách.
            text_mapping: Mapping thủ công (như cũ).
            ordinal_cols: Dictionary {tên_cột: [danh_sách_giá_trị_theo_thứ_tự_tăng_dần]}.
                          Ví dụ: {'education': ['No HS', 'HS', 'Bachelors', ...]}
            one_hot_cols: Danh sách các cột cần One-Hot Encoding.
        """
        super().__init__()
        self.datetime_col = datetime_col
        self.text_mapping = text_mapping
        self.ordinal_cols = ordinal_cols
        self.one_hot_cols = one_hot_cols
        # Lưu lại encoder để có thể inverse_transform nếu cần (cho ordinal)
        self._ordinal_encoders = {} 

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        try:
            # 1. Xử lý datetime (như cũ)
            if self.datetime_col and self.datetime_col in df_processed.columns:
                col = self.datetime_col
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                df_processed[f'{col}_year'] = df_processed[col].dt.year
                df_processed[f'{col}_month'] = df_processed[col].dt.month
                df_processed[f'{col}_day'] = df_processed[col].dt.day
                df_processed.drop(columns=[col], inplace=True)
                print(f"[FeatureEngineer] Đã tách ngày tháng cho cột '{col}'.")

            # 2. Xử lý mapping text thủ công (như cũ)
            if self.text_mapping:
                for col, mapping in self.text_mapping.items():
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].map(mapping).fillna(df_processed[col])
                        print(f"[FeatureEngineer] Đã map giá trị thủ công cho cột '{col}'.")

            # 3. Xử lý Ordinal Encoding (Mới)
            if self.ordinal_cols:
                df_processed = self._apply_ordinal_encoding(df_processed)

            # 4. Xử lý One-Hot Encoding (Mới)
            if self.one_hot_cols:
                df_processed = self._apply_one_hot_encoding(df_processed)

            self._update_state(df_processed)
            return df_processed
        except Exception as e:
            print(f"Lỗi trong FeatureEngineer: {e}")
            return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã hóa các biến có thứ tự."""
        for col, categories in self.ordinal_cols.items():
            if col in df.columns:
                # Scikit-learn OrdinalEncoder cần input là 2D array
                encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
                
                # Fit và transform
                encoded_data = encoder.fit_transform(df[[col]])
                
                # Cập nhật lại DataFrame (chuyển về 1D array)
                df[col] = encoded_data.flatten()
                
                # Lưu encoder
                self._ordinal_encoders[col] = encoder
                print(f"[FeatureEngineer] Đã Ordinal Encode cột '{col}' với thứ tự: {categories}")
            else:
                print(f"[FeatureEngineer] Cảnh báo: Cột '{col}' không tìm thấy để Ordinal Encode.")
        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã hóa One-Hot cho các biến danh định."""
        # Lọc các cột thực sự tồn tại
        valid_cols = [c for c in self.one_hot_cols if c in df.columns]
        
        if valid_cols:
            # Sử dụng pd.get_dummies cho đơn giản và hiệu quả với pandas DF
            # drop_first=True để tránh đa cộng tuyến (Dummy Variable Trap)
            df = pd.get_dummies(df, columns=valid_cols, dtype=int)
            print(f"[FeatureEngineer] Đã One-Hot Encode các cột: {valid_cols}")
        else:
            print("[FeatureEngineer] Không tìm thấy cột nào để One-Hot Encode.")
        return df


# =============================================================================
# 3. MANAGER CLASS (Lớp Quản Lý)
# =============================================================================

class DataManager:
    """
    Lớp quản lý dữ liệu, chấp nhận đầu vào là File Path hoặc DataFrame.
    """
    def __init__(self, input_data: Union[str, pd.DataFrame, None] = None) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.filepath: Optional[str] = None

        if input_data is not None:
            if isinstance(input_data, str):
                # Nếu là đường dẫn file string
                self.filepath = input_data
                self.load_data(input_data)
            elif isinstance(input_data, pd.DataFrame):
                # Nếu là DataFrame
                self.df = input_data.copy()
                print(f"--> [DataManager] Đã khởi tạo từ DataFrame có sẵn. Shape: {self.df.shape}")
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
            print(f"--> [DataManager] Đã tải dữ liệu từ file. Shape: {self.df.shape}")
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
        """Ghi dữ liệu sau xử lý ra file."""
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
    




# ... (Toàn bộ code Class ở trên giữ nguyên) ...

if __name__ == "__main__":
    import numpy as np
    
    print("=== DEMO PIPELINE XỬ LÝ DỮ LIỆU TỰ ĐỘNG ===")

    # 1. GIẢ LẬP DỮ LIỆU THÔ (RAW DATA)
    # Tạo dữ liệu chứa đủ các trường hợp: NaN, Outlier, Date, Category
    raw_data = {
        'date': ['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05'],
        'region': ['North', 'South', 'North', 'East', 'West'], # Biến phân loại
        'sales': [100.0, 120.0, np.nan, 110.0, 50000.0],       # NaN và Outlier cực lớn
        'temperature': [25.5, np.nan, np.nan, 26.0, 25.8],     # Dữ liệu liên tục bị thiếu
        'customer_rating': ['High', 'Low', 'Medium', 'High', 'Low'] # Biến thứ tự (Ordinal)
    }
    
    # Lưu tạm ra file CSV để test chức năng đọc file của DataManager
    df_raw = pd.DataFrame(raw_data)
    df_raw.to_csv('raw_data_demo.csv', index=False)
    
    print("-> Dữ liệu gốc (raw_data_demo.csv):")
    print(df_raw)
    print("-" * 50)

    # 2. KHỞI TẠO MANAGER VÀ LOAD DỮ LIỆU
    manager = DataManager('raw_data_demo.csv')

    # 3. ĐỊNH NGHĨA PIPELINE XỬ LÝ
    processing_steps = [
        # Bước 1: Xử lý giá trị thiếu
        # - 'sales': điền trung bình
        # - 'temperature': điền người trước đó (ffill) - Giả sử bạn đã thêm logic ffill vào Imputer
        Imputer(strategy='mean', columns=['sales']),
        Imputer(strategy='ffill', columns=['temperature']),

        # Bước 2: Feature Engineering
        # - Tách ngày tháng
        # - One-Hot cho 'region'
        # - Ordinal cho 'customer_rating'
        FeatureEngineer(
            datetime_col='date',
            one_hot_cols=['region'],
            ordinal_cols={'customer_rating': ['Low', 'Medium', 'High']}
        ),

        # Bước 3: Xử lý ngoại lai
        # - Loại bỏ giá trị 50000 trong cột sales bằng IQR
        OutlierHandler(method='iqr', action='remove', columns=['sales']),

        # Bước 4: Chuẩn hóa dữ liệu
        # - Đưa các cột số về khoảng [0, 1]
        Scaler(method='minmax')
    ]

    # 4. CHẠY PIPELINE
    for step in processing_steps:
        manager.apply(step)

    # 5. XUẤT KẾT QUẢ
    df_final = manager.get_data()
    print("-" * 50)
    print("-> Dữ liệu sau khi xử lý:")
    print(df_final)

    # 6. LƯU FILE KẾT QUẢ (Test hàm save_data mới thêm)
  
    if hasattr(manager, 'save_data'):
        manager.save_data('processed_data_final.csv')
    else:
        print("Cảnh báo: Bạn chưa thêm hàm save_data vào DataManager nên không lưu file.")