import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import box
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf

# Verilerin olduğu klasörler
images_dir = 'train/images'
labels_dir = 'train/labels'

def load_image(file_path):
    return cv2.imread(file_path)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_polygons(json_data):
    lng_lat_polygons = []
    xy_polygons = []
    
    for feature in json_data['features']['lng_lat']:
        lng_lat_polygons.append(wkt.loads(feature['wkt']))
    for feature in json_data['features']['xy']:
        xy_polygons.append(wkt.loads(feature['wkt']))
        
    return lng_lat_polygons, xy_polygons

def plot_image_with_polygons(image, polygons):
    plt.imshow(image)
    for poly in polygons:
        x, y = poly.exterior.xy
        plt.plot(x, y, color='red')
    plt.show()

# Model tanımı
def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

# Modeli derleme
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim verilerini hazırlama
X_train = []
Y_train = []

for image_name in os.listdir(images_dir):
    if 'pre' in image_name:  # Sadece pre-disaster görüntülerle çalışacağız
        image = load_image(os.path.join(images_dir, image_name))
        image = cv2.resize(image, (256, 256))  # Resimleri yeniden boyutlandır
        X_train.append(image)
        
        # Etiket dosyasını yükleme
        json_file = image_name.replace('png', 'json')
        json_data = load_json(os.path.join(labels_dir, json_file))
        _, xy_polygons = extract_polygons(json_data)
        
        # Maske oluşturma
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        for poly in xy_polygons:
            poly_coords = np.array(poly.exterior.coords, np.int32)
            cv2.fillPoly(mask, [poly_coords], 1)
        
        mask = cv2.resize(mask, (256, 256))  # Maskeyi yeniden boyutlandır
        Y_train.append(mask)

# Listeyi Numpy dizisine dönüştür
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = np.expand_dims(Y_train, axis=-1)  # Kanalları genişlet

# Örnek olarak ilk 50 görüntüyü alabiliriz
X_train = X_train[:50]
Y_train = Y_train[:50]

# Modeli eğitme
model.fit(X_train, Y_train, epochs=1, batch_size=8)

# Test görüntüsü üzerinde tahmin yapma ve görselleştirme
test_image_file = os.path.join(images_dir, 'guatemala-volcano_00000006_post_disaster.png')
test_image = load_image(test_image_file)
test_image_resized = cv2.resize(test_image, (256, 256))

predicted_mask = model.predict(np.expand_dims(test_image_resized, axis=0))[0]
predicted_mask = cv2.resize(predicted_mask, (1024, 1024))

# Tahmin edilen maskeyi kırmızı renkle görselleştirme
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Orijinal görüntüyü RGB formatına çevir

# Maskeyi threshold ile ikili hale getir
threshold = 0.3
predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)

# Maskeyi kırmızı bir katman olarak oluştur
red_mask = np.zeros_like(test_image_rgb)
red_mask[:, :, 0] = predicted_mask_binary * 255  # Kırmızı kanal (R)

# Orijinal görüntü ile kırmızı maskeyi birleştir (alpha blending)
alpha = 0.5  # Maskenin saydamlık değeri
overlay_image = cv2.addWeighted(test_image_rgb, 1, red_mask, alpha, 0)

# Görüntüyü göster
plt.figure(figsize=(8, 8))
plt.title("Binalar Kırmızı Renkte")
plt.imshow(overlay_image)
plt.axis('off')
plt.show()

# Morfolojik işlemler uygulama
kernel = np.ones((5, 5), np.uint8)
predicted_mask_binary = cv2.morphologyEx(predicted_mask_binary, cv2.MORPH_OPEN, kernel)
predicted_mask_binary = cv2.morphologyEx(predicted_mask_binary, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(8, 8))
plt.title("Binary Mask")
plt.imshow(predicted_mask_binary, cmap='gray')
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(predicted_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_circles = test_image.copy()
for contour in contours:
    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 5:  # Küçük daireleri göz ardı edebiliriz
        cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)  # Yeşil renkte çiz

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))
plt.title("Hasarlı Bölgeleri İşaretlenmiş Orijinal Görüntü")
plt.axis('off')
plt.show()

