import math
import os
import oracledb
import cv2 as cv
import numpy as np
from skimage.feature import hog

connection = oracledb.connect(
    user="system",
    password="hieu08052002",
    dsn="127.0.0.1:1521/orcl"
)
cursor = connection.cursor()

folder_path = 'C:\\Users\\untiv\\Desktop\\data\\Output_Image'


def main():
    # Lưu ảnh
    image = []

    #  Lưu vector đặc trưng
    feature_vectors = []

    #  Lưu vector của điểm gốc
    centroids = []

    # Trích xuất đặc trưng
    first_min = False

    # Lưu giá trị max cho mỗi phần tử thuộc vector đặc trưng
    max_value = np.zeros(57048, dtype=float)

    # Lưu giá trị min cho mỗi phần tử thuộc vector đặc trưng
    min_value = np.zeros(57048, dtype=float)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Thêm ảnh vào image
        with open(file_path, 'rb') as f:
            image.append(f.read())

        #  Trích đặc trưng
        feature = extract_features(file_path)
        feature_vectors.append(feature)

        # Kiểm tra xem có phải điểm gốc
        if 'centroid' in filename:
            centroids.append(feature)

        #  Lấy giá trị nhỏ nhất và lớn nhất
        for i in range(feature.size):
            if feature[i] > max_value[i]:
                max_value[i] = feature[i]
            if first_min == False or feature[i] < min_value[i]:
                min_value[i] = feature[i]
        first_min = True


    # Lưu max_value và min_value vào trong csdl
    data = ';'.join(str(num) for num in max_value)
    cursor.execute('insert into max_vector values(:data)', data=data)
    data = ';'.join(str(num) for num in min_value)
    cursor.execute('insert into min_vector values(:data)', data=data)
    connection.commit()


    # Chuẩn hóa vector đặc trưng
    for feature in feature_vectors:
        for i in range(feature.size):
            if min_value[i] == max_value[i]:
                feature[i] = 1
            else:
                feature[i] = round(float((feature[i]-min_value[i]) / (max_value[i]-min_value[i])), 2)

    # K-means
    cluster_id = np.zeros(len(feature_vectors))
    loop = 10
    while loop > 0:
        # Số lượng điểm mỗi cụm
        count = np.zeros(len(centroids))

        # Điểm gốc mới
        new_centroids = centroids

        # Tính khoảng cách mỗi điểm đến điểm gốc
        for i in range(len(feature_vectors)):
            cluster = 0
            distance = float('inf')
            for j in range(len(centroids)):
                d = calculate_distance(centroids[j], feature_vectors[i])
                if d < distance:
                    distance = d
                    cluster = j
            cluster_id[i] = cluster
            count[cluster] += 1

            # Chuẩn bị để tính lại điểm gốc
            new_centroids[cluster] = np.array(centroids[cluster]) + np.array(feature_vectors[i])

        # Tính toán lại điểm gốc
        for i in range(len(new_centroids)):
            new_centroids[i] = np.array(new_centroids[i]) / count[i]

        # Hết 1 vòng lặp
        loop -= 1

    # Lưu ảnh, mã cụm và vector đặc trưng vào bảng flowers
    for i in range(len(feature_vectors)):
        sql = 'insert into flowers (id, image, cluster_id, feature) values (:id, :image, :cluster_id, :feature)'
        feature = ';'.join(str(num) for num in feature_vectors[i])
        cursor.execute(sql, id=i, image=image[i], cluster_id=cluster_id[i], feature=feature)
    connection.commit()

    # Lưu vector của điểm gốc vào bảng centroids
    for i in range(len(centroids)):
        sql = 'insert into centroids (cluster_id, feature_vector) values (:cluster_id, :feature_vector)'
        feature_vector = ';'.join(str(num) for num in centroids[i])
        cursor.execute(sql, cluster_id=i, feature_vector=feature_vector)
    connection.commit()


# Hàm trích xuất đặc trưng
def extract_features(image_path):
    image = cv.imread(image_path)
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


if __name__ == "__main__":
    main()

