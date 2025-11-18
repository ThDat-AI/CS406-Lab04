# config.py
import os

# Cấu hình chung
IMG_SIZE = (150, 150)  # Kích thước ảnh đầu vào mặc định cho VGG16/MobileNet

# Danh sách các nhãn (Classes) - Sửa lại theo model của bạn (ví dụ: 6 class)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Đường dẫn đến các file model .h5
# Định dạng: "Tên hiển thị": "Đường dẫn file"
MODEL_PATHS = {
    "MobileNetV2": "models/MobileNetV2.h5",
    "VGG16": "models/VGG16.h5",
    "ResNet50": "models/ResNet50.h5"
}

# Cấu hình hiển thị biểu đồ
CHART_COLOR_HIGHLIGHT = "#2ecc71"  # Màu xanh lá cho model tốt nhất/nhanh nhất
CHART_COLOR_NORMAL = "#95a5a6"     # Màu xám cho các model còn lại