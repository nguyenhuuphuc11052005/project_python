# File dùng để chạy thử module preprocessing



import pandas as pd
import numpy as np
from preprocessing import DataManager, Imputer, Scaler, OutlierHandler, FeatureEngineer

if __name__ == "__main__":
    print("=== DEMO PIPELINE XỬ LÝ DỮ LIỆU TỰ ĐỘNG ===")

    # 1. GIẢ LẬP DỮ LIỆU THÔ
    raw_data = {
        'date': ['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05'],
        'region': ['North', 'South', 'North', 'East', 'West'],
        'sales': [100.0, 120.0, np.nan, 110.0, 50000.0],
        'temperature': [25.5, np.nan, np.nan, 26.0, 25.8],
        'customer_rating': ['High', 'Low', 'Medium', 'High', 'Low']
    }
    
    df_raw = pd.DataFrame(raw_data)
    df_raw.to_csv('raw_data_demo.csv', index=False)
    
    print("-> Dữ liệu gốc:")
    print(df_raw)
    print("-" * 50)

    # 2. KHỞI TẠO MANAGER
    manager = DataManager('raw_data_demo.csv')

    # 3. ĐỊNH NGHĨA PIPELINE
    processing_steps = [
        Imputer(strategy='mean', columns=['sales']),
        Imputer(strategy='ffill', columns=['temperature']),
        FeatureEngineer(
            datetime_col='date',
            one_hot_cols=['region'],
            ordinal_cols={'customer_rating': ['Low', 'Medium', 'High']}
        ),
        OutlierHandler(method='iqr', action='remove', columns=['sales']),
        Scaler(method='minmax')
    ]

    # 4. CHẠY PIPELINE
    for step in processing_steps:
        manager.apply(step)

    # 5. XUẤT KẾT QUẢ
    print("-" * 50)
    print("-> Dữ liệu sau khi xử lý:")
    print(manager.get_data())

    # 6. LƯU FILE
    manager.save_data('processed_data_final.csv')