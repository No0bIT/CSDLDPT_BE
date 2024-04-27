import base64
import io
import math
import oracledb
from flask import Flask
from skimage.feature import hog
import cv2 as cv
import numpy as np

connection = oracledb.connect(
    user="pvd",
    password="12345",
    dsn="localhost:1521/oracle"
)
cursor = connection.cursor()

app = Flask(__name__)
@app.route('/api')
def findImage(image):

    #  Trích xuất đặc trưng
    feature_vector = extract_features(image)

    #  Lấy giá trị lớn nhất
    cursor.execute('select * from max_vector')
    max_vector = cursor.fetchone()[0]
    max_vector = [float(i) for i in max_vector.read().split(';')]

    #  Lấy giá trị nhỏ nhất
    cursor.execute('select * from min_vector')
    min_vector = cursor.fetchone()[0]
    min_vector = [float(i) for i in min_vector.read().split(';')]

    # Chuẩn hóa vector đặc trưng
    for i in range(len(feature_vector)):
        if min_vector[i] == max_vector[i] or feature_vector[i] > max_vector[i]:
            feature_vector[i] = 1
        elif feature_vector[i] < min_vector[i]:
            feature_vector[i] = 0
        else:
            feature_vector[i] = round(float((feature_vector[i] - min_vector[i]) / (max_vector[i] - min_vector[i])), 2)


    # Lấy điểm gốc
    cursor.execute('select feature_vector from centroids')
    centroids = cursor.fetchall()

    #  Tính khoảng cách từ vector đặc trưng đến điểm gốc
    distance = float('inf')
    cluster_id = 0
    for i in range(len(centroids)):
        centroid = [float(k) for k in centroids[i][0].read().split(';')]
        d = calculate_distance(feature_vector, centroid)
        if d < distance:
            distance = d
            cluster_id = i

    #  Lấy tất cả ảnh thuộc cụm có mã bằng cluster_id
    cursor.execute('select * from flowers where cluster_id = :cluster_id', cluster_id=cluster_id)
    images = cursor.fetchall()

    # Tìm 3 ảnh giống nhất
    index = [0, 0, 0]
    distances = [float('inf'), float('inf'), float('inf')]
    for i in range(len(images)):
        feature = [float(k) for k in images[i][3].read().split(';')]
        d = calculate_distance(feature, feature_vector)

        k = 0
        while k < 3 and d > distances[k]:
            k += 1

        if k < 3:
            new_distance = d
            old_distance = distances[k]
            new_index = i
            old_index = index[k]
            while k < 2:
                # Cập nhật giá trị khoảng cách
                distances[k] = new_distance
                new_distance = old_distance
                old_distance = distances[k+1]

                # Cập nhật index của ảnh
                index[k] = new_index
                new_index = old_index
                old_index = index[k + 1]

                k += 1

            distances[k] = new_distance
            index[k] = new_index


    # Hiển thị kết quả cho test
    # for i in index:
    #     blob_data = images[i][1]
    #     # binary_data = base64.b64encode(blob_data.read())
    #     # data = binary_data.decode("UTF-8")
    #     stream = io.BytesIO(blob_data.read())
    #     image = cv.imdecode(np.frombuffer(stream.read(), np.uint8), cv.IMREAD_COLOR)
    #     cv.imshow('image', image)
    #     cv.waitKey(0)

    result = []
    for i in index:
        blob_data = images[i][1]
        stream = io.BytesIO(blob_data.read())
        image = cv.imdecode(np.frombuffer(stream.read(), np.uint8), cv.IMREAD_COLOR)
        result.append({image: image})
    return result


# Hàm trích xuất đặc trưng
def extract_features(image):
    hsv_vector = extract_hsv(image)
    hog_vector = extract_hog(image)
    return np.concatenate((hsv_vector, hog_vector))


# Trích xuất đặc trưng hsv
def extract_hsv(image):
    # Mỗi mảng lưu một giá trị thuộc hệ màu HSV
    h_vector = np.zeros(6, dtype=int) # Lưu giá trị Hue
    s_vector = np.zeros(8, dtype=int) # Lưu giá trị Saturation
    v_vector = np.zeros(10, dtype=int) # Lưu giá trị Value

    # Duyệt qua từng pixel trong ảnh
    width = image.shape[1]
    height = image.shape[0]
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            H, S, V = convert_bgr_to_hsv(R, G, B)

            # Chia bin ------------------------------
            h_index = min(5, math.floor(H / 60))
            s_index = min(5, math.floor(S / 0.125))
            v_index = min(5, math.floor(V / 0.1))

            h_vector[h_index] += 1
            s_vector[s_index] += 1
            v_vector[v_index] += 1
            # ---------------------------------------
    return np.concatenate((h_vector, s_vector, v_vector))


# Chuyển từ hệ màu BGR sang HSV
def convert_bgr_to_hsv(R, G, B):
    R, G, B = R / 255, G / 255, B / 255
    max_rgb = max(R, G, B)
    min_rgb = min(R, G, B)

    V = max_rgb
    S = 0
    if V != 0:
        S = (V - min_rgb) / V
    H = 0
    if R == G and G == B:
        H = 0
    elif V == R:
        H = 60 * (G - B) / (V - min_rgb)
    elif V == G:
        H = 120 + 60 * (B - R) / (V - min_rgb)
    elif V == B:
        H = 240 + 60 * (R - G) / (V - min_rgb)
    if H < 0:
        H = H + 360
    return [H, S, V]


# Trích đặc trưng hog
def extract_hog(image):
    gray_image = convert_bgr_to_gray(image)
    (hog_vector, hog_image) = hog(gray_image,
                                  orientations=9,
                                  pixels_per_cell=(8, 8),
                                  transform_sqrt=True,
                                  cells_per_block=(2, 2),
                                  block_norm="L2",
                                  visualize=True)
    return hog_vector

# Chuyển đồi hệ màu từ bgr sang hệ màu xám
def convert_bgr_to_gray(image):
    width = image.shape[1]
    height = image.shape[0]
    gray_image = np.zeros((height, width), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            gray_image[j, i] = 0.299 * R + 0.587 * G + 0.114 * B
    return gray_image


# Hàm tính toán khoảng cách Euclidean
def calculate_distance(centroid, new_image):
    result = 0
    for i in range(len(centroid)):
        result += (centroid[i] - new_image[i]) ** 2
    return math.sqrt(result)
