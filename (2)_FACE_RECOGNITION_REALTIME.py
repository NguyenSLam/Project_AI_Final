import cv2
import numpy as np
import time
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Tải mô hình đã được huấn luyện
model = load_model('FaceDetect_RealTime_final6.h5')

# Tải bộ phân loại Cascade của khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bản đồ ánh xạ các nhãn với các tên
label_map = {0: 'None', 1: 'Name:Thoa-ID:20146076', 2: 'Name:Lam-ID:20146362', 
             3: 'Name:Vu-ID:20146463', 4: 'Name:Nhat-ID:20146384', 5: 'Name:Viet-ID:20146216', 
             6: 'Name:Nguyen-ID:20146373', 7: 'Name:Phong-ID:20149078', 8: 'Name:Quyet-ID:20146408'}

# Tạo đối tượng VideoCapture và đặt độ phân giải
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200) #giảm kích thước khung hình
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)

# Tạo một placeholder cho luồng video
video_placeholder = st.empty()

# Tạo một placeholder cho bộ đếm fps
fps_placeholder = st.empty()

# Tốc độ khung hình mong muốn (fps)
desired_fps = 40

# Tạo một nút để dừng / bắt đầu luồng video
stop_button = st.button('Stop')

# Bắt đầu luồng video
while True:
    # Kiểm tra xem nút dừng đã được nhấn
    if stop_button:
        # Giải phóng VideoCapture và đóng cửa sổ
        cap.release()
        cv2.destroyAllWindows()
        
        # Đặtlại nút dừng
        stop_button = st.button('Start')
        
        # Nếu nút bắt đầu được nhấn, hãy thoát khỏi vòng lặp hiện tại và bắt đầu luồng video mới
        if stop_button:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)
            
            # Đặt lại các placeholder
            video_placeholder.empty()
            fps_placeholder.empty()
            
            # Thoát khỏi vòng lặp hiện tại và bắt đầu một vòng lặp mới
            break   
        
    # Bắt đầu đếm thời gian để đo FPS
    start_time = cv2.getTickCount()     
       
    # Đọc khung hình từ VideoCapture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển đổi khung hình sang xám
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    # Áp dụng cân bằng lược đồ màu để cải thiện độ tương phản
    mask = cv2.equalizeHist(mask)
    # Áp dụng median blur để giảm nhiễu
    mask = cv2.medianBlur(mask, 5)
    
    # Phát hiện khuôn mặt trong khung hình xám
    faces = face_cascade.detectMultiScale(mask, 1.3, 5)
    
    # Xử lý mỗi khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Trích xuất vùng khuôn mặt
        face = frame[y:y+h, x:x+w]
        
        # Thay đổi kích thước và chuẩn hóa ảnh khuôn mặt
        resized_face = cv2.resize(face,(100, 100))
        # Thay đổi kích thước thành một mảng NumPy
        normalized_face = img_to_array(resized_face)
        normalized_face = normalized_face.astype('float32')
        # Chuẩn hóa mảng NumPy bằng cách chia tất cả các giá trị trong mảng cho 255.0
        normalized_face = normalized_face / 255.0
        #Lật ngược khuôn mặt bằng phương thức flip của OpenCV để tăng khả năng phát hiện khuôn mặt 
        flipped_face = cv2.flip(normalized_face, 1)       
        reshaped_face = np.reshape(normalized_face, (1, 100, 100, 3))
        
        # Dự đoán nhãn của khuôn mặt
        result = model.predict(reshaped_face)
        label = np.argmax(result, axis=1)
        label_name = label_map[label[0]]
        
        # Vẽ hộp giới hạn và nhãn trên khung hình
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
        
        # Hiển thị thông tin bổ sung về khuôn mặt được phát hiện dựa trên bản đồ ánh xạ nhãn
        # có thể thêm thông tin bổ sung khác ở đây
        
    # Tính toán fps và hiển thị trên khung hình
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    fps = 1 / elapsed_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)   
    
    # Hiển thị khung hình đã xử lý trong ứng dụng Streamlit
    video_placeholder.image(frame, channels="BGR")
    
    # Cập nhật bộ đếm fps trong ứng dụng Streamlit
    fps_placeholder.text(f"FPS: {int(fps)}")    
    
    # Đợi để ổn định fps
    delay = 1 / desired_fps - elapsed_time
    if delay > 0:
        time.sleep(delay)    
        
    # Kiểm tra phím 'q' được nhấn để thoát khỏi vòng lặp vô hạn
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
        
# Giải phóng VideoCapture và đóng cửa sổ
cap.release() 
cv2.destroyAllWindows()