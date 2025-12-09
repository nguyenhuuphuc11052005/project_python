import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Optional

class DataVisualizer:
    """
    Class hỗ trợ trực quan hóa dữ liệu có tính tái sử dụng cao.
    
    Attributes:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        figsize (tuple): Kích thước mặc định của biểu đồ (width, height).
        palette (str): Bảng màu mặc định của Seaborn.
    """

    def __init__(self, df: pd.DataFrame, style: str = 'whitegrid', palette: str = 'viridis', figsize: tuple = (10, 6)):
        """
        Khởi tạo Visualizer.

        Args:
            df: DataFrame dữ liệu.
            style: Style nền của seaborn (whitegrid, darkgrid, ticks...).
            palette: Bảng màu (viridis, deep, muted...).
            figsize: Kích thước biểu đồ.
        """
        self.df = df
        self.figsize = figsize
        self.palette = palette
        
        # Thiết lập cấu hình toàn cục
        sns.set_theme(style=style, palette=palette)
        # Sửa font tiếng Việt nếu cần (tùy môi trường)
        plt.rcParams['axes.unicode_minus'] = False 

    def _finalize_plot(self, title: str, xlabel: str = None, ylabel: str = None, save_path: str = None):
        """
        Hàm nội bộ (Protected) để xử lý các bước cuối cùng của biểu đồ:
        Tiêu đề, nhãn, hiển thị và lưu file.
        """
        if title: plt.title(title, fontsize=14, fontweight='bold', pad=15)
        if xlabel: plt.xlabel(xlabel, fontsize=12)
        if ylabel: plt.ylabel(ylabel, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            # Tự động tạo thư mục nếu chưa có
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ tại: {save_path}")
        
        plt.show()
        plt.close() # Đóng plot để giải phóng bộ nhớ

    # --- CÁC PHƯƠNG THỨC VẼ ---

    def plot_histogram(self, column: str, bins: int = 30, kde: bool = True, 
                       title: str = None, save_path: str = None, **kwargs):
        """
        Vẽ biểu đồ phân phối (Histogram) cho biến số.
        
        Args:
            **kwargs: Các tham số khác của sns.histplot (vd: hue, element...)
        """
        plt.figure(figsize=self.figsize)
        sns.histplot(data=self.df, x=column, bins=bins, kde=kde, **kwargs)
        
        self._finalize_plot(
            title=title or f'Distribution of {column}',
            xlabel=column,
            ylabel='Frequency',
            save_path=save_path
        )

    def plot_bar(self, x: str, y: str = None, estimator=None, 
                 title: str = None, save_path: str = None, **kwargs):
        """
        Vẽ biểu đồ cột (Barplot) hoặc đếm (Countplot).
        
        Nếu `y` là None -> Vẽ Countplot (đếm số lượng).
        Nếu `y` có giá trị -> Vẽ Barplot (giá trị trung bình hoặc tổng).
        """
        plt.figure(figsize=self.figsize)
        
        if y is None:
            # Count plot
            sns.countplot(data=self.df, x=x, **kwargs)
            ylabel_text = 'Count'
        else:
            # Bar plot (Aggregated)
            sns.barplot(data=self.df, x=x, y=y, estimator=estimator or sum, errorbar=None, **kwargs)
            ylabel_text = y

        self._finalize_plot(
            title=title or f'Bar Chart of {x}',
            xlabel=x,
            ylabel=ylabel_text,
            save_path=save_path
        )

    def plot_box(self, x: str, y: str, title: str = None, save_path: str = None, **kwargs):
        """
        Vẽ biểu đồ hộp (Boxplot) để xem phân phối và ngoại lai.
        """
        plt.figure(figsize=self.figsize)
        sns.boxplot(data=self.df, x=x, y=y, **kwargs)
        
        self._finalize_plot(
            title=title or f'Boxplot of {y} by {x}',
            xlabel=x,
            ylabel=y,
            save_path=save_path
        )

    def plot_scatter(self, x: str, y: str, hue: str = None, 
                     title: str = None, save_path: str = None, **kwargs):
        """
        Vẽ biểu đồ phân tán (Scatterplot) để xem tương quan.
        """
        plt.figure(figsize=self.figsize)
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, style=hue, s=100, alpha=0.8, **kwargs)
        
        self._finalize_plot(
            title=title or f'Scatter Plot: {x} vs {y}',
            xlabel=x,
            ylabel=y,
            save_path=save_path
        )

    def plot_heatmap(self, columns: List[str] = None, title: str = None, save_path: str = None, **kwargs):
        """
        Vẽ biểu đồ nhiệt (Heatmap) thể hiện ma trận tương quan.
        
        Args:
            columns: Danh sách các cột số cần tính tương quan. Nếu None, lấy tất cả cột số.
        """
        plt.figure(figsize=self.figsize) 
        
        if columns:
            corr_matrix = self.df[columns].corr()
        else:
            corr_matrix = self.df.select_dtypes(include='number').corr()
            
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, **kwargs)
        
        self._finalize_plot(
            title=title or 'Correlation Heatmap',
            save_path=save_path
        )

    def plot_line(self, x: str, y: str, hue: str = None, 
                  title: str = None, save_path: str = None, **kwargs):
        """Vẽ biểu đồ đường (Lineplot) cho chuỗi thời gian."""
        plt.figure(figsize=self.figsize)
        sns.lineplot(data=self.df, x=x, y=y, hue=hue, marker='o', **kwargs)
        
        self._finalize_plot(
            title=title or f'Trend of {y} over {x}',
            xlabel=x,
            ylabel=y,
            save_path=save_path
        )





if __name__ == "__main__":
    import numpy as np

    # 1. TẠO DỮ LIỆU GIẢ LẬP
    print("--- Đang tạo dữ liệu mẫu... ---")
    np.random.seed(42)  # Để kết quả cố định
    n_samples = 100
    
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
        'Department': np.random.choice(['Sales', 'IT', 'HR', 'Marketing'], n_samples),
        'Age': np.random.randint(22, 60, n_samples),
        'Salary': np.random.normal(1500, 500, n_samples).round(2), # Lương phân phối chuẩn
        'Performance_Score': np.random.uniform(1, 10, n_samples).round(1),
        'Years_Experience': np.random.randint(1, 20, n_samples)
    }
    
    # Tạo mối tương quan giả: Kinh nghiệm càng cao lương càng cao
    data['Salary'] += data['Years_Experience'] * 100 
    
    df_dummy = pd.DataFrame(data)
    print(f"Dữ liệu mẫu shape: {df_dummy.shape}")
    print(df_dummy.head())

    # 2. KHỞI TẠO VISUALIZER
    print("\n--- Khởi tạo DataVisualizer ---")
    # Sử dụng style 'whitegrid' và palette 'viridis'
    viz = DataVisualizer(df_dummy, style='whitegrid', palette='viridis', figsize=(10, 6))

    # Tạo thư mục output để test tính năng lưu file
    output_dir = "test_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. TEST CÁC BIỂU ĐỒ

    # A. Histogram: Phân phối lương
    print("1. Vẽ Histogram...")
    viz.plot_histogram(
        column='Salary', 
        bins=20, 
        title='Phân phối Lương nhân viên',
        save_path=f"{output_dir}/1_salary_hist.png"
    )

    # B. Countplot: Số lượng nhân viên mỗi phòng ban
    print("2. Vẽ Countplot...")
    viz.plot_bar(
        x='Department', 
        title='Số lượng nhân viên theo phòng ban',
        save_path=f"{output_dir}/2_dept_count.png"
    )

    # C. Barplot (Aggregated): Lương trung bình theo phòng ban
    print("3. Vẽ Barplot (Mean Salary)...")
    viz.plot_bar(
        x='Department', 
        y='Salary', 
        estimator=np.mean, 
        title='Lương trung bình theo phòng ban',
        save_path=f"{output_dir}/3_dept_salary_mean.png"
    )

    # D. Boxplot: Phân phối tuổi theo phòng ban
    print("4. Vẽ Boxplot...")
    viz.plot_box(
        x='Department', 
        y='Age', 
        title='Phân phối tuổi theo phòng ban',
        save_path=f"{output_dir}/4_age_box.png"
    )

    # E. Scatterplot: Tương quan giữa Kinh nghiệm và Lương (có màu theo Phòng ban)
    print("5. Vẽ Scatterplot...")
    viz.plot_scatter(
        x='Years_Experience', 
        y='Salary', 
        hue='Department', 
        title='Tương quan: Kinh nghiệm vs Lương',
        save_path=f"{output_dir}/5_exp_salary_scatter.png"
    )

    # F. Heatmap: Ma trận tương quan các biến số
    print("6. Vẽ Heatmap...")
    viz.plot_heatmap(
        columns=['Age', 'Salary', 'Performance_Score', 'Years_Experience'], 
        title='Ma trận tương quan',
        save_path=f"{output_dir}/6_correlation.png"
    )

    # G. Lineplot: Xu hướng lương theo thời gian (giả lập)
    # Tạo dataframe mới gom nhóm theo tháng để vẽ line cho đẹp
    print("7. Vẽ Lineplot...")
    df_dummy['Month'] = df_dummy['Date'].dt.to_period('M').astype(str)
    df_trend = df_dummy.groupby('Month')['Salary'].mean().reset_index()
    
    # Khởi tạo viz mới cho dữ liệu trend
    viz_trend = DataVisualizer(df_trend)
    viz_trend.plot_line(
        x='Month', 
        y='Salary', 
        title='Xu hướng lương trung bình theo tháng',
        save_path=f"{output_dir}/7_salary_trend.png"
    )

    print(f"\n✅ Hoàn tất! Vui lòng kiểm tra thư mục '{output_dir}' để xem các file ảnh đã lưu.")