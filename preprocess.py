import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy import stats

class DataPreprocessor:
    """
    Một lớp Python toàn diện để thực hiện các bước tiền xử lý dữ liệu,
    bao gồm đọc, làm sạch, chuẩn hóa, mã hóa và trích xuất đặc trưng.

    Thuộc tính:
        df (pd.DataFrame): DataFrame chứa dữ liệu đang được xử lý.
        original_df (pd.DataFrame): Một bản sao của DataFrame gốc trước khi
                                    thực hiện bất kỳ thay đổi nào.
    """

    def __init__(self, dataframe: pd.DataFrame = None):
        """
        Khởi tạo đối tượng DataPreprocessor.

        Args:
            dataframe (pd.DataFrame, optional): Một DataFrame để tải vào
                                                bộ xử lý khi khởi tạo.
        """
        if dataframe is not None:
            self.df = dataframe.copy()
            self.original_df = dataframe.copy()
        else:
            self.df = None
            self.original_df = None

    def __repr__(self) -> str:
        """
        Trả về một biểu diễn chuỗi của đối tượng, hiển thị trạng thái của DataFrame.
        """
        if self.df is None:
            return "DataPreprocessor(Empty: Chưa có dữ liệu)"
        
        shape = self.df.shape
        cols = self.df.columns.tolist()
        return f"DataPreprocessor(Shape: {shape}, Columns: {cols})"

    # --- 1. Đọc và Ghi dữ liệu ---

    @classmethod
    def from_file(cls, filepath: str):
        """
        Một phương thức của lớp (@classmethod) để tạo một đối tượng
        DataPreprocessor trực tiếp từ một đường dẫn file.

        Args:
            filepath (str): Đường dẫn đến file (csv, xlsx, json).

        Returns:
            DataPreprocessor: Một đối tượng mới của lớp này với dữ liệu đã được tải.
        """
        instance = cls()
        instance.load_data(filepath)
        return instance

    def load_data(self, filepath: str):
        """
        Đọc dữ liệu từ file (csv, xlsx, json) vào DataFrame của đối tượng.

        Xử lý lỗi FileNotFoundError và các lỗi đọc file chung.

        Args:
            filepath (str): Đường dẫn đến file.
        """
        try:
            _, extension = os.path.splitext(filepath)
            
            if extension == '.csv':
                self.df = pd.read_csv(filepath)
            elif extension == '.xlsx':
                self.df = pd.read_excel(filepath)
            elif extension == '.json':
                self.df = pd.read_json(filepath)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {extension}")
            
            self.original_df = self.df.copy()
            print(f"Tải dữ liệu thành công từ: {filepath}")

        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file tại '{filepath}'")
            self.df = None
            self.original_df = None
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            self.df = None
            self.original_df = None

    def save_data(self, filepath: str, index: bool = False):
        """
        Lưu DataFrame đã xử lý ra file mới.

        Args:
            filepath (str): Đường dẫn file đầu ra.
            index (bool, optional): Có lưu chỉ số (index) của DataFrame không. Mặc định là False.
        """
        if self.df is None:
            print("Lỗi: Không có dữ liệu để lưu.")
            return

        try:
            _, extension = os.path.splitext(filepath)
            
            if extension == '.csv':
                self.df.to_csv(filepath, index=index)
            elif extension == '.xlsx':
                self.df.to_excel(filepath, index=index)
            elif extension == '.json':
                self.df.to_json(filepath, orient='records', indent=4)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {extension}")
            
            print(f"Lưu dữ liệu đã xử lý thành công tại: {filepath}")

        except Exception as e:
            print(f"Lỗi khi lưu file: {e}")

    def get_processed_data(self) -> pd.DataFrame:
        """
        Trả về DataFrame đã được xử lý.

        Returns:
            pd.DataFrame: DataFrame hiện tại.
        """
        return self.df

    # --- 2. Hàm tiện ích và Tự động phát hiện ---

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> tuple[list, list, list]:
        """
        (staticmethod) Tự động phát hiện và phân loại các cột
        thành 3 nhóm: số, phân loại (object/category), và ngày giờ.

        Args:
            df (pd.DataFrame): DataFrame để phân tích.

        Returns:
            tuple[list, list, list]: Một tuple chứa 
                                     (numeric_cols, categorical_cols, datetime_cols)
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        
        # Lọc ra các cột datetime tiềm năng khỏi danh sách categorical
        potential_dt_cols = []
        remaining_categorical_cols = []
        
        for col in categorical_cols:
            try:
                # Thử parse một vài giá trị non-null để xác nhận
                pd.to_datetime(df[col].dropna().sample(min(5, len(df[col].dropna()))), errors='raise')
                # Nếu thành công, đây có thể là cột datetime
                if col not in datetime_cols:
                    potential_dt_cols.append(col)
            except Exception:
                # Nếu thất bại, nó là categorical
                remaining_categorical_cols.append(col)
        
        # Cảnh báo người dùng về các cột datetime tiềm năng chưa được chuyển đổi
        if potential_dt_cols:
            print(f"Cảnh báo: Các cột sau có vẻ là datetime nhưng đang ở dạng 'object': {potential_dt_cols}")
            print("Hãy sử dụng pd.to_datetime() trước khi dùng engineer_datetime_features().")

        # Trả về các cột đã được xác nhận
        return numeric_cols, remaining_categorical_cols, datetime_cols

    # --- 3. Xử lý dữ liệu thiếu (Missing Data) ---

    def handle_missing_values(self, strategy: str = 'mean', subset: list = None, fill_value=None):
        """
        Xử lý giá trị bị thiếu (NaN) trong các cột được chỉ định hoặc toàn bộ DataFrame.

        Chiến lược 'auto' sẽ tự động áp dụng 'mean' cho cột số và 'mode' cho cột phân loại.

        Args:
            strategy (str): Chiến lược điền giá trị:
                'mean', 'median': Chỉ áp dụng cho cột số.
                'mode': Áp dụng cho cột số hoặc phân loại.
                'ffill': Forward fill.
                'bfill': Backward fill.
                'constant': Điền bằng giá trị `fill_value`.
                'drop': Xóa các hàng có giá trị thiếu.
                'auto': Tự động dùng 'mean' cho số, 'mode' cho phân loại.
            subset (list, optional): Danh sách các cột để áp dụng.
                                     Nếu None, áp dụng cho các cột phù hợp.
            fill_value (any, optional): Giá trị để điền nếu strategy='constant'.
        """
        if self.df is None:
            print("Lỗi: Vui lòng tải dữ liệu trước.")
            return

        if strategy == 'drop':
            self.df.dropna(subset=subset, inplace=True)
            print(f"Đã xóa các hàng có giá trị NaN (trong subset: {subset}).")
            return

        numeric_cols, categorical_cols, _ = self.get_column_types(self.df)
        
        if subset is None:
            if strategy == 'auto':
                # Tự động xử lý tất cả các cột
                self._fill_missing(numeric_cols, 'mean')
                self._fill_missing(categorical_cols, 'mode')
            elif strategy in ['mean', 'median']:
                self._fill_missing(numeric_cols, strategy)
            else:
                self._fill_missing(self.df.columns, strategy, fill_value)
        else:
            # Chỉ xử lý các cột trong subset
            self._fill_missing(subset, strategy, fill_value)

    def _fill_missing(self, columns: list, strategy: str, fill_value=None):
        """Hàm trợ giúp nội bộ cho handle_missing_values."""
        try:
            for col in columns:
                if col not in self.df.columns:
                    print(f"Cảnh báo: Cột '{col}' không tồn tại. Bỏ qua.")
                    continue
                
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'ffill':
                    self.df[col].ffill(inplace=True)
                    continue # ffill/bfill không cần fill_val
                elif strategy == 'bfill':
                    self.df[col].bfill(inplace=True)
                    continue
                elif strategy == 'constant':
                    fill_val = fill_value
                else:
                    print(f"Chiến lược '{strategy}' không hợp lệ. Bỏ qua cột {col}.")
                    continue
                
                self.df[col].fillna(fill_val, inplace=True)
        except Exception as e:
            print(f"Lỗi khi điền giá trị thiếu cho cột {columns} với chiến lược {strategy}: {e}")


    # --- 4. Xử lý dữ liệu ngoại lai (Outliers) ---

    def handle_outliers(self, method: str = 'iqr', action: str = 'remove', threshold: float = 1.5, columns: list = None):
        """
        Phát hiện và xử lý dữ liệu ngoại lai bằng IQR, Z-score, hoặc Isolation Forest.

        Args:
            method (str): Phương pháp: 'iqr', 'zscore', 'isolation_forest'.
            action (str): Hành động: 'remove' (xóa hàng) hoặc 'cap' (thay thế bằng
                          giá trị giới hạn - chỉ hỗ trợ 'iqr' và 'zscore').
            threshold (float): Ngưỡng:
                - cho 'iqr': Hệ số nhân (mặc định 1.5).
                - cho 'zscore': Giá trị Z-score (mặc định 3.0).
                - cho 'isolation_forest': Tỷ lệ contamination (mặc định 0.1).
            columns (list, optional): Các cột số để kiểm tra. Nếu None,
                                      tự động chọn tất cả các cột số.
        """
        if self.df is None: return

        if columns is None:
            columns, _, _ = self.get_column_types(self.df)
        
        original_rows = self.df.shape[0]

        try:
            if method == 'iqr':
                for col in columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    if action == 'remove':
                        self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    elif action == 'cap':
                        self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
            
            elif method == 'zscore':
                if threshold is None: threshold = 3.0 # Đặt ngưỡng mặc định cho z-score
                for col in columns:
                    z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                    if action == 'remove':
                        # Lấy chỉ số của các giá trị không ngoại lai
                        non_outlier_indices = self.df[col].dropna()[z_scores < threshold].index
                        # Giữ lại các hàng NaN và các hàng không ngoại lai
                        self.df = self.df[self.df[col].isna() | self.df.index.isin(non_outlier_indices)]
                    elif action == 'cap':
                        mean = self.df[col].mean()
                        std = self.df[col].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)

            elif method == 'isolation_forest':
                if action == 'cap':
                    print("Cảnh báo: 'cap' không được hỗ trợ cho 'isolation_forest'. Sử dụng 'remove'.")
                
                # Isolation Forest xử lý NaN không tốt, cần điền trước
                df_subset = self.df[columns].fillna(self.df[columns].mean())
                
                if threshold is None: threshold = 0.1 # contamination
                clf = IsolationForest(contamination=threshold, random_state=42)
                preds = clf.fit_predict(df_subset)
                
                # Giữ lại các giá trị là inliers (preds == 1)
                self.df = self.df[preds == 1]

            print(f"Xử lý ngoại lai (Phương pháp: {method}): {original_rows - self.df.shape[0]} hàng đã bị xóa/xử lý.")
            
        except Exception as e:
            print(f"Lỗi khi xử lý ngoại lai: {e}")

    # --- 5. Chuẩn hóa và Mã hóa (Scaling & Encoding) ---

    def scale_features(self, method: str = 'standard', columns: list = None):
        """
        Chuẩn hóa các đặc trưng số bằng StandardScaler hoặc MinMaxScaler.

        Args:
            method (str): 'standard' (StandardScaler) hoặc 'minmax' (MinMaxScaler).
            columns (list, optional): Các cột để chuẩn hóa. Nếu None,
                                      chuẩn hóa tất cả các cột số.
        """
        if self.df is None: return

        if columns is None:
            columns, _, _ = self.get_column_types(self.df)
        
        if not columns:
            print("Không tìm thấy cột số nào để chuẩn hóa.")
            return

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"Phương pháp chuẩn hóa '{method}' không hợp lệ. Chỉ hỗ trợ 'standard', 'minmax'.")
            return

        try:
            # Chỉ fit và transform trên các cột không thiếu
            valid_data = self.df[columns].dropna()
            if valid_data.empty:
                print(f"Không có dữ liệu hợp lệ trong các cột {columns} để chuẩn hóa.")
                return
                
            scaler.fit(valid_data)
            self.df[columns] = scaler.transform(self.df[columns])
            print(f"Đã chuẩn hóa các cột: {columns} bằng {method}")
        except Exception as e:
            print(f"Lỗi khi chuẩn hóa dữ liệu: {e}")

    def encode_categorical(self, method: str = 'onehot', columns: list = None, drop_first: bool = True):
        """
        Mã hóa các biến phân loại bằng One-Hot Encoding hoặc Label Encoding.

        Args:
            method (str): 'onehot' (sử dụng pd.get_dummies) hoặc 'label' (LabelEncoder).
            columns (list, optional): Cột để mã hóa. Nếu None, mã hóa tất cả
                                      các cột 'object' hoặc 'category'.
            drop_first (bool): Có xóa cột đầu tiên trong One-Hot Encoding
                               để tránh đa cộng tuyến. Mặc định là True.
        """
        if self.df is None: return

        if columns is None:
            _, columns, _ = self.get_column_types(self.df)
        
        if not columns:
            print("Không tìm thấy cột phân loại nào để mã hóa.")
            return

        try:
            if method == 'onehot':
                self.df = pd.get_dummies(self.df, columns=columns, drop_first=drop_first)
                print(f"Đã mã hóa One-Hot cho các cột: {columns}")
            elif method == 'label':
                le = LabelEncoder()
                for col in columns:
                    # LabelEncoder không xử lý tốt NaN, chuyển NaN sang 1 chuỗi riêng
                    self.df[col] = self.df[col].astype(str).fillna('NaN_String')
                    self.df[col] = le.fit_transform(self.df[col])
                print(f"Đã mã hóa Label cho các cột: {columns}")
            else:
                print(f"Phương pháp mã hóa '{method}' không hợp lệ. Chỉ hỗ trợ 'onehot', 'label'.")
        except Exception as e:
            print(f"Lỗi khi mã hóa dữ liệu: {e}")

    # --- 6. Trích xuất đặc trưng (Feature Engineering) ---

    @staticmethod
    def custom_text_to_numeric(series: pd.Series, mapping: dict) -> pd.Series:
        """
        (staticmethod) Một hàm tiện ích tự định nghĩa để chuyển đổi
        text sang số dựa trên một bản đồ (mapping)
        
        Ví dụ: {'Low': 1, 'Medium': 2, 'High': 3}

        Args:
            series (pd.Series): Cột dữ liệu dạng text.
            mapping (dict): Từ điển định nghĩa việc chuyển đổi.

        Returns:
            pd.Series: Cột dữ liệu đã được chuyển đổi.
        """
        return series.map(mapping)

    def apply_custom_mapping(self, column: str, mapping: dict):
        """
        Áp dụng hàm custom_text_to_numeric cho một cột cụ thể.

        Args:
            column (str): Tên cột cần áp dụng.
            mapping (dict): Từ điển định nghĩa việc chuyển đổi.
        """
        if self.df is None or column not in self.df.columns:
            print(f"Lỗi: Cột '{column}' không tồn tại.")
            return
        
        try:
            self.df[column] = self.custom_text_to_numeric(self.df[column], mapping)
            print(f"Đã áp dụng mapping tùy chỉnh cho cột '{column}'.")
        except Exception as e:
            print(f"Lỗi khi áp dụng mapping: {e}")

    def engineer_datetime_features(self, column: str, drop_original: bool = True):
        """
        Tạo các đặc trưng mới từ một cột ngày giờ (datetime).

        Các đặc trưng bao gồm: year, month, day, dayofweek, is_weekend.

        Args:
            column (str): Tên cột chứa dữ liệu ngày giờ.
            drop_original (bool): Có xóa cột ngày giờ gốc sau khi trích xuất.
                                  Mặc định là True.
        """
        if self.df is None or column not in self.df.columns:
            print(f"Lỗi: Cột '{column}' không tồn tại.")
            return

        try:
            # Cố gắng chuyển đổi nếu chưa đúng định dạng
            self.df[column] = pd.to_datetime(self.df[column])

            # Trích xuất đặc trưng
            self.df[f'{column}_year'] = self.df[column].dt.year
            self.df[f'{column}_month'] = self.df[column].dt.month
            self.df[f'{column}_day'] = self.df[column].dt.day
            self.df[f'{column}_dayofweek'] = self.df[column].dt.dayofweek
            self.df[f'{column}_is_weekend'] = (self.df[column].dt.dayofweek >= 5).astype(int)
            
            if drop_original:
                self.df.drop(column, axis=1, inplace=True)
                
            print(f"Đã trích xuất đặc trưng datetime từ cột '{column}'.")
            
        except Exception as e:
            print(f"Lỗi khi xử lý cột datetime '{column}': {e}")



# --- Chuẩn bị dữ liệu mẫu ---
data = {
    'ngay_mua': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-10'],
    'doanh_thu': [100, 110, np.nan, 105, 5000], # 5000 là outlier
    'chi_phi': [50, 55, 60, 52, 65],
    'thanh_pho': ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hà Nội', 'TP.HCM'],
    'xep_hang': ['Tốt', 'Rất Tốt', 'Tốt', 'Khá', np.nan] # 'NaN' là missing
}
sample_df = pd.DataFrame(data)

# --- 1. Khởi tạo đối tượng ---
# Bạn có thể khởi tạo bằng DataFrame...
processor = DataPreprocessor(sample_df)

# ...hoặc tải từ file (ví dụ nếu bạn đã lưu sample_df.to_csv('sample.csv'))
# processor = DataPreprocessor.from_file('sample.csv')

print("--- Dữ liệu gốc ---")
print(processor.get_processed_data())
print(processor) # Sử dụng __repr__

# --- 2. Xử lý dữ liệu thiếu (Missing Values) ---
# Tự động điền 'mean' cho 'doanh_thu' và 'mode' cho 'xep_hang'
processor.handle_missing_values(strategy='auto')
print("\n--- Sau khi xử lý NaN (auto) ---")
print(processor.get_processed_data())

# --- 3. Xử lý ngoại lai (Outliers) ---
# Xóa các hàng có 'doanh_thu' ngoại lai bằng IQR
processor.handle_outliers(method='iqr', action='remove', columns=['doanh_thu'])
print("\n--- Sau khi xử lý Outlier (IQR) ---")
print(processor.get_processed_data())

# --- 4. Trích xuất đặc trưng (Feature Engineering) ---
# 4a. Xử lý Datetime
processor.engineer_datetime_features('ngay_mua')
print("\n--- Sau khi trích xuất Datetime ---")
print(processor.get_processed_data().head())

# 4b. Xử lý Text tùy chỉnh (Custom Mapping)
mapping = {'Khá': 1, 'Tốt': 2, 'Rất Tốt': 3}
processor.apply_custom_mapping('xep_hang', mapping)
print("\n--- Sau khi Mapping tùy chỉnh 'xep_hang' ---")
print(processor.get_processed_data())

# --- 5. Mã hóa biến phân loại (Encoding) ---
# 'thanh_pho' là cột phân loại còn lại
processor.encode_categorical(method='onehot', columns=['thanh_pho'])
print("\n--- Sau khi Mã hóa One-Hot 'thanh_pho' ---")
print(processor.get_processed_data())

# --- 6. Chuẩn hóa dữ liệu (Scaling) ---
# Chuẩn hóa các cột số
numeric_cols_to_scale = ['doanh_thu', 'chi_phi', 'xep_hang']
processor.scale_features(method='standard', columns=numeric_cols_to_scale)
print("\n--- Dữ liệu cuối cùng sau khi chuẩn hóa ---")
print(processor.get_processed_data())

# --- 7. Lấy DataFrame cuối cùng hoặc lưu ra file ---
final_df = processor.get_processed_data()
# processor.save_data('processed_data.csv')