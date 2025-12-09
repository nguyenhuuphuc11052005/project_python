import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohankrishnathalla/medical-insurance-cost-prediction")

print("Path to dataset files:", path)