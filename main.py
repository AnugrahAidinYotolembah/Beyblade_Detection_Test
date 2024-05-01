import cv2
import numpy as np
import tensorflow as tf
import csv
import time
import os

# Definisi area arena (misalnya menggunakan koordinat x, y, lebar, dan tinggi)
arena_x, arena_y, arena_width, arena_height = 100, 100, 1024, 768

# Inisialisasi penghitung gasing
spinning_tops_count = 0
non_spinning_tops_count = 0

# Fungsi untuk mendeteksi kelas objek (berputar, tidak berputar, normal)
def detect_class(image, model):
    input_image = cv2.resize(image, (224, 224))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0

    predictions = model.predict(input_image)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Return class index based on the model's output
    if class_index in [0, 1]:
        return class_index, confidence
    elif class_index == 2:  # Adjust the index if non-spinning top class is different
        return 2, confidence
    else:
        return None, None

# Load model deteksi objek
model_path = '/Users/didiyotolembah19gmail.com/Documents/kerja/kecilin startup/kecilin test/model/keras_model.h5'
model = tf.keras.models.load_model(model_path)

# Inisialisasi video capture
video_path = '/Users/didiyotolembah19gmail.com/Documents/kerja/kecilin startup/kecilin test/video/video_input.mp4'
cap = cv2.VideoCapture(video_path)

# Inisialisasi video output
output_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video_path = 'output_video.mp4'
output_video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), output_frame_fps, (output_frame_width, output_frame_height))

# Buka file CSV untuk report
csv_file = open('detection_report.csv', mode='w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Class', 'accuracy', 'Bounding Box'])

# Inisialisasi variabel untuk menghitung durasi pertandingan
start_time = time.time()

# Fungsi untuk menggambar bounding box pada frame
def draw_bounding_box(frame, class_index, confidence, box, angle):
    global spinning_tops_count, non_spinning_tops_count

    # Cek apakah gasing berada di dalam arena
    center = np.array(box).mean(axis=0)
    if arena_x <= center[0] <= arena_x + arena_width and arena_y <= center[1] <= arena_y + arena_height:
        if class_index == 0:
            color = (0, 255, 0)  # Green for spinning top
            text = "Spinning Top"
            spinning_tops_count += 1
        elif class_index == 1:
            color = (0, 0, 255)  # Red for non-spinning top
            text = "Non-Spinning Top"
            non_spinning_tops_count += 1
        
        # Rotasi koordinat bounding box
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        corners = np.array([[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]]], dtype=np.float32)
        rotated_corners = cv2.transform(np.array([corners]), rot_mat)[0]
        rotated_x, rotated_y, rotated_w, rotated_h = cv2.boundingRect(rotated_corners.astype(int))

        # Simpan informasi ke dalam file CSV
        csv_writer.writerow([cap.get(cv2.CAP_PROP_POS_FRAMES), text, confidence, (rotated_x, rotated_y, rotated_w, rotated_h)])

        # Gambar bounding box pada frame yang sudah diputar
        cv2.rectangle(frame, (rotated_x, rotated_y), (rotated_x + rotated_w, rotated_y + rotated_h), color, 2)
        cv2.putText(frame, f"{text} ({confidence:.2f})", (rotated_x, rotated_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Loop untuk membaca frame-frame dari video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mendeteksi kelas objek pada setiap frame
    class_index, confidence = detect_class(frame, model)

    # Jika ada gasing di arena, lakukan deteksi dan gambar bounding box
    if class_index in [0, 1]:
        # Mendapatkan wilayah warna gasing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 100, 100])
        upper_bound = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Menghilangkan noise dengan opening
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Mendapatkan kontur gasing
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop untuk menggambar bounding box pada setiap gasing yang terdeteksi
        for contour in contours:
            # Mendapatkan sudut perputaran dari kontur gasing
            rect = cv2.minAreaRect(contour)
            angle = rect[2]

            # Mendapatkan koordinat bounding box
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Memfilter kotak yang kecil
            if cv2.contourArea(contour) > 100:
                draw_bounding_box(frame, class_index, confidence, box, angle)

    # Menyimpan frame yang sudah dimodifikasi ke video output
    output_video_writer.write(frame)

    # Menampilkan frame yang sudah dimodifikasi
    cv2.imshow('Object Detection', frame)

    # Exit jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menghitung dan menyimpan durasi pertandingan ke dalam file CSV
end_time = time.time()
duration = end_time - start_time
csv_writer.writerow(['Duration', '', '', duration])

# Menutup file CSV, video capture, dan semua jendela OpenCV
csv_file.close()
cap.release()
output_video_writer.release()
cv2.destroyAllWindows()

# Menampilkan pesan bahwa proses telah selesai
print("Video output berhasil disimpan:", output_video_path)
print(f"Jumlah gasing yang berputar: {spinning_tops_count}")
print(f"Jumlah gasing yang tidak berputar: {non_spinning_tops_count}")