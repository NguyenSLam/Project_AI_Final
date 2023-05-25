import cv2
import numpy as np
import streamlit as st
import base64
from keras.models import load_model
from keras.preprocessing.image import img_to_array
#-------------------------backgroud---------------------------------
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("bc.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 125%;
background-position: 30% 45%;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: 50% 45%;
background-size: 174%;
}}
[data-testid="stSidebarNav"] span {{
color:white;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
#----------------------------------------------------------------
import cv2
import numpy as np
import streamlit as st
import base64
from keras.models import load_model
from keras.preprocessing.image import img_to_array
# Tải mô hình đã được huấn luyện
model = load_model('FaceDetect_RealTime_final6.h5')
# Tải bộ phân loại Cascade của khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Bản đồ ánh xạ giữa nhãn và thông tin cá nhân
personal_info_map = {
    0: {'Name': 'None', 'ID': 'None', 'birthdate': 'None', 'major': 'None'},
    1: {'Name': '**Kim Thoa**', 'ID': '**20146076**', 'birthdate': '**16/02/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    2: {'Name': '**Son Lam**', 'ID': '**20146362**', 'birthdate': '**09/06/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    3: {'Name': '**Tan Vu**', 'ID': '**20146463**',  'birthdate': '**19/04/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    4: {'Name': '**Quang Nhat**', 'ID': '**20146384**', 'birthdate': '**01/01/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    5: {'Name': '**Anh Viet**', 'ID': '**20146216**', 'birthdate': '**18/09/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    6: {'Name': '**Nghia Nguyen**', 'ID': '**20146373**', 'birthdate':'**05/05/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    7: {'Name': '**Quoc Phong**', 'ID': '**20149078**', 'birthdate': '**26/06/2002**',
        'class':'**20146CL1A**', 'major': '**Mechatronics Engineering Technology**'},
    8: {'Name': '**Van Quyet**', 'ID': '**20146408**', 'birthdate': '**04/02/2002**',
        'class':'**20146CL1B**', 'major': '**Mechatronics Engineering Technology**'}
}
# Tiêu đề cho ứng dụng và trình tải tệp lên
st.title(":red[FACE RECOGNITION]😀")
# Tiêu đề cho ứng dụng và trình tải tệp lên
st.header("Facial recognition and student information provision🧑‍🎓")
# Tải tệp tin lên Streamlit
uploaded_file= st.file_uploader("Choose an image...📂", type=["jpg", "jpeg", "png","bmp"])
# Thực hiện nhận diện khuôn mặt khi nhấn nút "Start face recognition"
if st.button('Start face recognition🔍'):
    # Kiểm tra xem đã tải tệp lên chưa
    if uploaded_file is not None:
        st.write("Done!✅")
        # Đọc tệp ảnh
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        # Thay đổi kích thước ảnh
        resized_image = cv2.resize(image, (112, 112))
        # Chuyển đổi ảnh sang màu xám
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # Áp dụng cân bằng lược đồ màu để cải thiện độ tương phản
        gray = cv2.equalizeHist(gray)
        # Áp dụng lọc trung bình để giảm nhiễu
        gray = cv2.medianBlur(gray, 5)
        # Nhận diện khuôn mặt trong ảnh xám
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Xử lý từng khuôn mặt được nhận diện
        for (x, y, w, h) in faces:
            # Trích xuất vùng khuôn mặt
            face = resized_image[y:y+h, x:x+w]
            # Thay đổi kích thước và chuẩn hóa ảnh khuôn mặt
            resized_face = cv2.resize(face,(100, 100))
            # thay đổi kích thước thành một mảng NumPy
            normalized_face = img_to_array(resized_face)
            # Chuẩn hóa mảng NumPy bằng cách chia tất cả các giá trị trong mảng cho 255.0
            normalized_face = normalized_face.astype('float32')
            normalized_face = normalized_face / 255.0
            #Lật ngược khuôn mặt bằng phương thức flip của OpenCV để tăng khả năng phát hiện khuôn mặt            
            flipped_face = cv2.flip(normalized_face, 1)
            reshaped_face = np.reshape(normalized_face, (1,100, 100,3))
            # Dự đoán nhãn của khuôn mặt sử dụng mô hình đã huấn luyện
            prediction = model.predict(reshaped_face)
            label = np.argmax(prediction)
             # Lấy thông tin cá nhân của nhãn được dự đoán
            personal_info = personal_info_map[label]
            # Tạo văn bản để hiển thị
            text = f"<br>Name: {personal_info['Name']}<br>ID: {personal_info['ID']}<br>DoB: {personal_info['birthdate']}<br>Class: {personal_info['class']}<br>Major: {personal_info['major']}"
            # Vẽ hình chữ nhật xung quanh khuôn mặt được nhận diện
            cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Hiển thị thông tin cá nhân của nhãn được dự đoán trên giao diện Streamlit
            st.write("The detected face belongs to:", text, unsafe_allow_html=True)
        # Hiển thị ảnh đã được xử lý trên giao diện Streamlit
        st.image(resized_image, channels="BGR")