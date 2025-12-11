import logging
import sys

def setup_logging(log_file: str = "training.log"):
    """
    Cấu hình hệ thống logging (Phiên bản Robust cho VS Code/Local).
    """
    # 1. Lấy Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 2. Xóa sạch các handlers cũ (để tránh in lặp hoặc bị chặn)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. Định nghĩa Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 4. Handler 1: Ghi ra File (với encoding utf-8 để không lỗi font)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 5. Handler 2: Ghi ra Terminal (sys.stdout)
    # Quan trọng: Dùng sys.stdout và flush để đẩy log ra ngay
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    # Test ngay lập tức xem có in được không
    logging.info("✅ Hệ thống Logging đã được cấu hình thành công trên VS Code.")