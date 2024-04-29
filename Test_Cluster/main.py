import base64
import io

import oracledb
import matplotlib.pyplot as plt
from PIL import Image

connection = oracledb.connect(
    user="vmh",
    password="Hieu08052002",
    dsn="localhost:1521/oracle"
)

cursor = connection.cursor()
cursor.execute('select * from flowers where cluster_id = :id', id=3)
results = cursor.fetchall()
for row in results:
    image_result = row[1]
    binary_data = base64.b64decode(image_result.read())
    image = Image.open(io.BytesIO(binary_data))
    # data = binary_data.decode("UTF-8")
    # plt.imshow(data)
    # plt.show()