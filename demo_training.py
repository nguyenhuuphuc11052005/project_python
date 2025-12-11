# File dùng để chạy demo module model_traing


import pandas as pd
import argparse
from model_training import ModelTrainer, setup_logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

if __name__ == "__main__":
    # 1. Cấu hình logging
    setup_logging()

    # 2. Xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'auto'], default='train')
    # Thêm tham số chọn scaler từ dòng lệnh
    parser.add_argument('--scaler', type=str, choices=['standard', 'minmax', 'robust', 'none'], default='standard')
    args = parser.parse_args()

    # 3. Tạo dữ liệu giả (Dummy Data)
    print("--- Đang tạo dữ liệu mẫu ---")
    X_dummy, y_dummy = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_dummy[:, 0] = X_dummy[:, 0] * 1000  
    X_dummy[:, 1] = X_dummy[:, 1] * 0.001 
    
    df = pd.DataFrame(X_dummy, columns=[f'feat_{i}' for i in range(10)])
    df['target'] = y_dummy
    
    # 4. Khởi tạo Trainer
    trainer = ModelTrainer(random_state=42, task_type='classification')
    
    # 5. Load & Split
    trainer.load_data(df.drop(columns=['target']), df['target'])
    trainer.split_data()
    
    # --- BƯỚC GỌI scale_data() CŨ ĐÃ BỊ XÓA ---

    # 6. Chạy theo mode (Truyền loại scaler vào đây)
    if args.mode == 'train':
        print(f"\n=== MODE: SINGLE TRAIN (SVM) with {args.scaler} scaler ===")
        svm = SVC()
        params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        
        # Truyền scaler_type vào hàm optimize_params
        trainer.optimize_params(svm, params, method='random', scaler_type=args.scaler)
        
        trainer.evaluate()
        trainer.save_model("final_model.pkl")

    elif args.mode == 'auto':
        print(f"\n=== MODE: AUTO TRAIN (RACE) with {args.scaler} scaler ===")
        models = {
            'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100]}),
            'SVM': (SVC(), {'C': [1, 10]})
        }
        # Truyền scaler_type vào hàm auto_train
        trainer.auto_train(models, scaler_type=args.scaler)
        
        trainer.evaluate()