import numpy as np
import time
import base64
import streamlit as st
from keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array

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
# Load model (absolute path)
model = load_model("FaceDetect_image_final.h5")
# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def main():
    # Title the web
    st.title(":red[FACE RECOGNITION]")
    # Display the header and "Drag and drop" bar
    st.header("Facial recognition and student information provision🧑‍🎓")
    st.write(":warning: Note about the image:") 
    st.write(":heavy_check_mark: _SHOULD be smooth image_")
    st.write(":heavy_check_mark: _MUST show the direct object_")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png","bmp"])
    
    # Process the file (if true)
    if uploaded_file is not None:
        successMessage = st.empty()
        successMessage = st.success("Upload image successfully!", icon="✅")
        # Create button
        click = st.button("Click here to detect right now!")
        img_PIL = Image.open(uploaded_file)
        img_array = np.array(img_PIL) # np.array -> RGB

        if(click):
            successMessage.empty()
            # status element: spinner
            with st.spinner('Loading...'):
                time.sleep(2)
            st.success('Done!')

            # Resize the image to the desired dimensions
            resized_img = cv2.resize(img_array, (200,200))
            photo = img_to_array(resized_img)
            photo= photo.astype('float32')
            photo = photo / 255
            photo = np.expand_dims(photo, axis=0)

            result = model.predict(photo).argmax()

            class_name = ['Unknown', 'THOA', 'HUNG', 'QUYET', 'PHONG', 'QUAN', 'LAM', 'NHAT', 'VU', 'VIET', 'NHA']
            
            if class_name[result] == 'THOA':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146076**', 'name': '**Nguyễn Thị Kim Thoa**','major':'**Mechatronics Engineering Technology**','date of birth':'**16/02/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'HUNG':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146344**', 'name': '**Trần Thảo Hưng**','major':'**Mechatronics Engineering Technology**','date of birth':'**13/12/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'QUYET':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146408**', 'name': '**Nguyễn Văn Quyết**','major':'**Mechatronics Engineering Technology**','date of birth':'**04/02/2002**','class':'**20146CL1B**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'PHONG':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20149078**', 'name': '**Huỳnh Quốc Phong**','major':'**Mechatronics Engineering Technology**','date of birth':'**26/06/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'QUAN':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146403**', 'name': '**Nguyễn Toàn Quân**','major':'**Mechatronics Engineering Technology**','date of birth':'**12/04/2002**','class':'**20146CL1B**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'LAM':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146362**', 'name': '**Nguyễn Sơn Lâm**','major':'**Mechatronics Engineering Technology**','date of birth':'**09/06/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'NHAT':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146384**', 'name': '**Nguyễn Đoàn Quang Nhật**','major':'**Mechatronics Engineering Technology**','date of birth':'**01/01/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'VU':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146463**', 'name': '**Phan Tấn Vũ**','major':'**Mechatronics Engineering Technology**','date of birth':'**19/04/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'VIET':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146216**', 'name': '**Lộ Anh Việt**','major':'**Mechatronics Engineering Technology**','date of birth':'**18/09/2002**','class':'**20146CL1A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            elif class_name[result] == 'NHA':
                # Tìm kiếm thông tin của sinh viên tương ứng
                student_info = {'mssv': '**20146377**', 'name': '**Đặng Bảo Nha**','major':'**Mechatronics Engineering Technology**','date of birth':'**09/07/2002**','class':'**20146CL5A**'}
                st.write(f"- Full name: {student_info['name']}")
                st.write(f"- ID: {student_info['mssv']}")
                st.write(f"- Major: {student_info['major']}")
                st.write(f"- Class: {student_info['class']}")
                st.write(f"- DoB: {student_info['date of birth']}")
            else:
                st.write(class_name[result])
            
            plt.imshow(img_array)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
if __name__ == "__main__":
    main()