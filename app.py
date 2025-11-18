import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time
import pandas as pd
import altair as alt
import config  # File config chứa MODEL_PATHS (bao gồm cả path ResNet), CLASS_NAMES, IMG_SIZE

# --- IMPORT HÀM PREPROCESS CHO TỪNG LOẠI MODEL ---
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import resnet50

# Cấu hình trang
st.set_page_config(page_title="Model Inference Demo", layout="centered")

st.title("🔍 Demo Phân Loại Ảnh")

# --- 1. HÀM LOAD MODEL (HỖ TRỢ: MOBILENET, VGG, RESNET) ---
st.sidebar.header("Trạng thái Model")
loaded_models = {}

@st.cache_resource
def load_model_info(name, path):
    """
    Load model và tự động chọn đúng hàm preprocess_input.
    Hỗ trợ: MobileNetV2, VGG16, ResNet50
    """
    # Chuẩn hóa tên và đường dẫn để so sánh
    name_lower = name.lower()
    path_lower = path.lower()

    # 1. Xác định hàm preprocess dựa trên loại model
    if "vgg" in name_lower or "vgg" in path_lower:
        # VGG: Trừ mean, BGR
        func_preprocess = vgg16.preprocess_input
        # print(f"Log: {name} -> VGG mode")
        
    elif "resnet" in name_lower or "resnet" in path_lower:
        # ResNet: Trừ mean, BGR (tương tự VGG nhưng khác thông số mean)
        func_preprocess = resnet50.preprocess_input
        # print(f"Log: {name} -> ResNet mode")
        
    else:
        # Mặc định là MobileNetV2: Scale về [-1, 1]
        func_preprocess = mobilenet_v2.preprocess_input
        # print(f"Log: {name} -> MobileNet mode")
    
    # 2. Tạo custom_objects
    # Map nhiều key để đề phòng model lưu tên hàm khác nhau
    custom_objects = {
        "preprocess_input": func_preprocess,
        "resnet_preprocess": func_preprocess, # Fix trường hợp lưu tên biến là resnet_preprocess
        "vgg_preprocess": func_preprocess     # Fix trường hợp lưu tên biến là vgg_preprocess
    }
    
    try:
        # Thử load với custom_objects
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    except Exception as e1:
        try:
            # Fallback: Thử load với safe_mode=False
            return tf.keras.models.load_model(path, custom_objects=custom_objects, safe_mode=False)
        except Exception as e2:
            raise e1

# Load toàn bộ model khi app khởi động
for name, path in config.MODEL_PATHS.items():
    try:
        model = load_model_info(name, path)
        loaded_models[name] = model
        st.sidebar.success(f"✅ {name} sẵn sàng")
    except Exception as e:
        st.sidebar.error(f"❌ {name} lỗi: {str(e)}")

# --- 2. HÀM XỬ LÝ ẢNH (GIỮ NGUYÊN 0-255) ---
def preprocess_image(image, target_size):
    """
    Chuẩn bị ảnh cho model.
    Output: Tensor (1, H, W, 3) với giá trị pixel [0, 255]
    Lý do: Tất cả các model (MobileNet, VGG, ResNet) đều đã có lớp Lambda 
           bên trong để tự xử lý (chia hoặc trừ mean) từ input gốc.
    """
    # 1. Chuyển sang RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize ảnh (LANCZOS cho chất lượng tốt nhất)
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # 3. Chuyển sang mảng numpy
    img_array = np.array(image)
    
    # 4. Thêm chiều batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Cast về float32 nhưng KHÔNG chia 255
    img_array = img_array.astype('float32')
    
    return img_array

# --- 3. GIAO DIỆN CHÍNH ---
uploaded_file = st.file_uploader("Chọn ảnh để phân loại (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- HIỂN THỊ ẢNH ---
    st.subheader("1. Ảnh đầu vào")
    image = Image.open(uploaded_file)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(image, caption='Input Image', use_container_width=True)
    
    st.markdown("---") 

    # Tiền xử lý
    processed_img = preprocess_image(image, config.IMG_SIZE)

    if not loaded_models:
        st.warning("⚠️ Không có model nào được load thành công.")
    else:
        results = []
        
        # ==================================================================
        # BƯỚC WARM-UP (LÀM NÓNG ENGINE)
        # ==================================================================
        # Chạy nháp để TensorFlow nạp thư viện CUDA và khởi tạo Graph.
        # Đảm bảo tính thời gian công bằng cho tất cả model.
        with st.spinner("Đang khởi động engine & Warm-up models..."):
            for _, model in loaded_models.items():
                model.predict(processed_img, verbose=0)
        
        # ==================================================================
        # BẮT ĐẦU ĐO THỜI GIAN THỰC
        # ==================================================================
        for model_name, model in loaded_models.items():
            # Dùng perf_counter cho độ chính xác cao (micro-second)
            start_time = time.perf_counter()
            
            predictions = model.predict(processed_img, verbose=0)
            
            end_time = time.perf_counter()
            inf_time = end_time - start_time
            
            # Xử lý output
            if predictions.shape[1] > 1:
                idx = np.argmax(predictions[0])
                conf = np.max(predictions[0]) * 100
                label = config.CLASS_NAMES[idx] if idx < len(config.CLASS_NAMES) else f"Class {idx}"
            else:
                score = predictions[0][0]
                conf = score * 100 if score > 0.5 else (1 - score) * 100
                label = config.CLASS_NAMES[1] if score > 0.5 else config.CLASS_NAMES[0]
            
            results.append({
                "Tên Model": model_name,
                "Dự đoán": label,
                "Độ tin cậy (%)": round(conf, 2),
                "Thời gian (s)": round(inf_time, 4)
            })
        
        # Tạo DataFrame
        df = pd.DataFrame(results)
        df.insert(0, 'STT', range(1, 1 + len(df)))

        # --- HIỂN THỊ KẾT QUẢ ---
        st.subheader("2. Kết quả chi tiết")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")

        # Logic tô màu biểu đồ (Highlight Best Performance)
        if not df.empty:
            best_conf = df["Độ tin cậy (%)"].max()
            best_time = df["Thời gian (s)"].min()

            df['color_conf'] = df['Độ tin cậy (%)'].apply(
                lambda x: config.CHART_COLOR_HIGHLIGHT if x == best_conf else config.CHART_COLOR_NORMAL
            )
            df['color_time'] = df['Thời gian (s)'].apply(
                lambda x: config.CHART_COLOR_HIGHLIGHT if x == best_time else config.CHART_COLOR_NORMAL
            )

            # Biểu đồ Confidence
            st.subheader("3. So sánh độ tin cậy (Confidence)")
            chart_conf = alt.Chart(df).mark_bar().encode(
                x=alt.X('Tên Model', axis=alt.Axis(labelAngle=0)),
                y='Độ tin cậy (%)',
                color=alt.Color('color_conf', scale=None),
                tooltip=['Tên Model', 'Dự đoán', 'Độ tin cậy (%)']
            ).properties(height=300)
            st.altair_chart(chart_conf, use_container_width=True)

            st.markdown("---")

            # Biểu đồ Inference Time
            st.subheader("4. So sánh tốc độ (Inference Time)")
            chart_time = alt.Chart(df).mark_bar().encode(
                x=alt.X('Tên Model', axis=alt.Axis(labelAngle=0)),
                y='Thời gian (s)',
                color=alt.Color('color_time', scale=None),
                tooltip=['Tên Model', 'Thời gian (s)']
            ).properties(height=300)
            st.altair_chart(chart_time, use_container_width=True)