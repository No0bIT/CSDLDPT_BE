from math import sqrt
import os
from PIL import Image

# Đường dẫn tới thư mục chứa các ảnh gốc của bạn
data_dir = "H:\DAT\HCSDL_DPT\Data"
# Đường dẫn đến thư mục để lưu các ảnh đã chỉnh sửa
output_dir = "H:\DAT\HCSDL_DPT\Test_Image"

def calculate_non_white_ratio(image):
    non_white_pixels = sum(1 for pixel in image.getdata() if pixel != (255, 255, 255))
    total_pixels = image.width * image.height
    ratio = non_white_pixels / total_pixels
    return ratio

def add_white_pixels(image, target_ratio):
    ratio = calculate_non_white_ratio(image)
    pxImage = ratio*360*300
    size = int(pxImage/target_ratio) 
    width_diff = int(sqrt(size * 5 / 6))
    height_diff =   int(size/width_diff)
    new_image = Image.new("RGB", (width_diff,height_diff), (255, 255, 255)) 
    new_image.paste(image, ((width_diff - image.width) // 2, (height_diff - image.height) // 2))
    ratio = calculate_non_white_ratio(new_image)

    new_image = new_image.resize(base_image.size)
    ratio = calculate_non_white_ratio(new_image)
    print(f"ratio: {ratio}")
    
    print()
    return new_image

    
    




# Kiểm tra xem thư mục đích có tồn tại không, nếu không thì tạo mới
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".jpg") or filename.endswith(".png")]
images = []

# Mở và thêm các ảnh vào danh sách images
for path in image_paths:
    image = Image.open(path)
    images.append(image)
# Chọn ảnh có tỷ lệ pixel không phải màu trắng trên tổng số pixel màu trắng là nhỏ nhất
base_image = min(images, key=calculate_non_white_ratio)
base_ratio = calculate_non_white_ratio(base_image)

print(base_ratio)
# Thêm các pixel màu trắng vào các ảnh khác để có tỉ lệ tương tự ảnh cơ sở
adjusted_images = []
for image in images[1:]:
    image.resize((300,360))
    adjusted_image = add_white_pixels(image, base_ratio)  
    adjusted_images.append(adjusted_image)

# Lưu các ảnh đã chỉnh sửa vào thư mục output_dir
for i, image in enumerate(adjusted_images):
    filename = f"hoa_{i+1}__{image.size}.jpg"
    image.save(os.path.join(output_dir, filename))
