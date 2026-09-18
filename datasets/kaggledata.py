# pyrefly: ignore [missing-import]
import kagglehub

# Download latest version
path = kagglehub.dataset_download("benjamin1717/taitanic")

print("Path to dataset files:", path)