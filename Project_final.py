import base64
import plotly.express as px
import streamlit as st
import numpy as np
import cv2
import time

df = px.data.iris()

#@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("bc.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 150%;
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
st.write("# Chào Mừng bạn đến với website Artificial intelligence của Nguyễn Sơn Lâm ! 👋")
with st.sidebar: 
     st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
     st.sidebar.success("**Trên đây là dự án của tôi☝️.**")
st.markdown(
"""
    **Xin chào Thầy và các bạn! Cảm ơn vì đã ghé thăm website của tôi.Ở đây là dự án tôi học được qua môn Artificial intelligence từ thầy Nguyễn Trường Thịnh.
    Website này được tạo ra nhằm mục đích báo cáo project cuối kì cho môn học trên và chỉ dùng để cho phục vụ cho học tập.**
    
    **👈 Click vào đây để xem dự án mà tôi đã được học ở bên trái.**
    ### Thông tin cá nhân:
    - **Họ và tên: Nguyễn Sơn Lâm**
    - **Mã số sinh viên: 20146362**
    - **Trường: Đại học Sư phạm Kỹ thuật TP.Hồ Chí Minh**
    ### Liên hệ với tôi:
    - Github: https://github.com/NguyenSLam
    - Facebook: https://www.facebook.com/profile.php?id=100091991723543
    - Email: nguyensonlam9602@gmail.com
    - Phone: 0389630921 🤙
    ### Thông tin giảng viên hướng dẫn:
    - **Thầy: PGS.TS Nguyễn Trường Thịnh**
    - Email: openlab@hcmute.edu.vn 
    - Github: https://github.com/Truongthinhs 
"""
)
#------Tao Contact Form-------
with st.container():
    st.write("---")
    st.header("TƯƠNG TÁC, Đánh GIÁ và NHẬN XÉT VỚI MÌNH TẠI ĐÂY'💁‍♂️'")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/nguyensonlam9602@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Tên của bạn" required>
     <input type="email" name="email"placeholder=" Email của bạn" required>
     <textarea name = "message" placeholder=" Đóng góp ý kiến tại đây nhé!" required></textarea>
     <button type="submit">Send</button>
</form>
"""
left_column, right_column = st.columns(2)
with left_column:
    st.markdown(contact_form, unsafe_allow_html=True)
with right_column:
    st.empty()
# YouTube video URL
with st.container():
    st.write("---")
    st.header("Video youtube dự án cuối kỳ AI của mình'💁‍♂️'")
youtube_url = 'https://youtu.be/4mGsA7X5WVw'

# Display video
st.video(youtube_url)








