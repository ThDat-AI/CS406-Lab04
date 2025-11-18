# AI Model Comparison Demo - Streamlit App

Ứng dụng Streamlit để demo và so sánh hai model AI: **MobileNet2** và **VGG16**

## 🚀 Cài đặt & Chạy

### 1. Cài đặt Dependencies
\`\`\`bash
cd streamlit_app
pip install -r requirements.txt
\`\`\`

### 2. Chuẩn bị Model Files
Tạo thư mục `models` và đặt file model:
\`\`\`
streamlit_app/
├── models/
│   ├── mobilenet2.h5
│   └── vgg16.h5
├── app.py
├── config.py
├── utils.py
├── requirements.txt
└── README.md
\`\`\`

### 3. Chạy Ứng Dụng
\`\`\`bash
streamlit run app.py
\`\`\`

Ứng dụng sẽ mở trên browser tại `http://localhost:8501`

## 📊 Tính Năng

✅ **Sidebar Status**: Hiển thị trạng thái load model (tick xanh khi load thành công)

✅ **Image Upload**: Cho phép người dùng upload ảnh

✅ **Model Inference**: Chạy inference với từng model (load 1 lần 1 model)

✅ **Results Table**: Bảng kết quả gồm:
- STT (số thứ tự)
- Tên Model
- Predicted Class
- Confidence (%)
- Inference Time (s)

✅ **Confidence Chart**: Biểu đồ cột so sánh confidence, hiển thị model tốt nhất

✅ **Inference Time Chart**: Biểu đồ cột so sánh tốc độ inference

✅ **Summary Metrics**: Tóm tắt kết quả (Best, Average)

## ⚙️ Cấu Hình

Chỉnh sửa `config.py` để:
- Thêm/xóa model
- Thay đổi đường dẫn file model
- Cấu hình input size
- Thay đổi các tham số khác

## 🎨 UI/UX

- Giao diện sạch sẽ, dễ sử dụng
- Responsive design phù hợp với Streamlit
- Biểu đồ tương tác
- Status indicator rõ ràng

## 📝 Lưu Ý

- Mỗi lần inference chỉ load 1 model để tối ưu hóa bộ nhớ
- Support các định dạng: jpg, jpeg, png, bmp, gif
- Có xử lý error chi tiết
