from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import io
import base64
import time
from werkzeug.utils import secure_filename

# Model yükleme için gerekli kütüphaneler
import tensorflow as tf

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Eğitilmiş model yolu - bu yolu kendi modelinizin yoluna göre ayarlayın
MODEL_PATH = 'trained_model.h5'

# Eğitilmiş modeli yükleme fonksiyonu
def load_model():
    """
    Eğitilmiş modeli yükler
    Bu fonksiyonu kullanacağınız modele göre değiştirin
    """
    try:
        # Burada kendi modelinizi yükleme kodunuzu ekleyin
        model = tf.keras.models.load_model(MODEL_PATH)
        
        print("Model başarıyla yüklendi")
        return model
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        return None

# Uygulama başladığında modeli yükle
model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Eğitilmiş model ile hasar analizi
def analyze_damage(before_image_path, after_image_path):
    """
    Eğitilmiş model ile before ve after görüntüleri arasındaki hasarı analiz eder
    """
    try:
        # Görüntüleri yükle
        before_img = Image.open(before_image_path).convert('RGB')
        after_img = Image.open(after_image_path).convert('RGB')
        
        # Görüntüleri aynı boyuta getir (gerekirse)
        if before_img.size != after_img.size:
            after_img = after_img.resize(before_img.size)
        
        # Görüntüleri modelin beklediği formata dönüştür
        before_tensor = tf.convert_to_tensor(np.array(before_img) / 255.0)
        after_tensor = tf.convert_to_tensor(np.array(after_img) / 255.0)
        
        # Model tahminini yap
        prediction = model.predict(tf.stack([before_tensor, after_tensor], axis=0))
        
        # Hasar oranını hesapla
        damage_percentage = int(prediction[0][0] * 100)
        
        # Fark görselleştirmesi oluştur
        diff_img = Image.fromarray(np.abs(np.array(after_img).astype(np.int32) - np.array(before_img).astype(np.int32)).astype(np.uint8))
        diff_path = os.path.join(app.config['UPLOAD_FOLDER'], f'diff_{time.time()}.jpg')
        diff_img.save(diff_path)
        
        # Hasar şiddetini belirle
        if damage_percentage < 20:
            severity = 'Hafif'
        elif damage_percentage < 50:
            severity = 'Orta'
        elif damage_percentage < 75:
            severity = 'Orta-Ağır'
        else:
            severity = 'Ağır'
        
        # Şiddete göre öneriler oluştur
        recommendations = []
        if severity == 'Hafif':
            recommendations = [
                'Yapısal olmayan hasarların tamiri',
                'Düzenli kontrol önerilir'
            ]
        elif severity == 'Orta':
            recommendations = [
                'Yapısal değerlendirme gerekli',
                'Uzman mühendis incelemesi önerilir',
                'Hasarlı bölgelerin takviyesi gerekebilir'
            ]
        elif severity == 'Orta-Ağır':
            recommendations = [
                'Kapsamlı yapısal değerlendirme gerekli',
                'Güçlendirme çalışması önerilir',
                'Kullanım kısıtlaması gerekebilir'
            ]
        else:  # Ağır
            recommendations = [
                'Acil yapısal müdahale gerekli',
                'Bina tahliyesi değerlendirilmeli',
                'Kapsamlı yenileme veya yıkım gerekebilir'
            ]
        
        return {
            'success': True,
            'damagePercentage': damage_percentage,
            'severity': severity,
            'recommendations': recommendations,
            'processedImages': {
                'difference': diff_path,
                'highlighted': diff_path  
            }
        }
    
    except Exception as e:
        print(f"Görüntü analiz hatası: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'beforeImage' not in request.files or 'afterImage' not in request.files:
        return jsonify({'error': 'Her iki görüntü de gerekli'}), 400
    
    before_file = request.files['beforeImage']
    after_file = request.files['afterImage']
    
    if before_file.filename == '' or after_file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not (allowed_file(before_file.filename) and allowed_file(after_file.filename)):
        return jsonify({'error': 'Dosya türü desteklenmiyor'}), 400
    
    # Dosyaları kaydet
    before_filename = secure_filename(f"before_{time.time()}_{before_file.filename}")
    after_filename = secure_filename(f"after_{time.time()}_{after_file.filename}")
    
    before_path = os.path.join(app.config['UPLOAD_FOLDER'], before_filename)
    after_path = os.path.join(app.config['UPLOAD_FOLDER'], after_filename)
    
    before_file.save(before_path)
    after_file.save(after_path)
    
    # Görüntüleri analiz et
    result = analyze_damage(before_path, after_path)
    
    if not result['success']:
        return jsonify({'error': result['error']}), 500
    
    # Analiz sonucunu döndür
    return jsonify({
        'results': {
            'damagePercentage': result['damagePercentage'],
            'severity': result['severity'],
            'recommendations': result['recommendations'],
            'processedImages': result['processedImages']
        }
    })

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Model durumunu kontrol etmek için endpoint"""
    return jsonify({
        'status': 'ready' if model is not None else 'not_loaded',
        'message': 'Model hazır' if model is not None else 'Model yüklenemedi'
    })

@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """Modeli yeniden yüklemek için endpoint"""
    global model
    model = load_model()
    return jsonify({
        'status': 'success' if model is not None else 'error',
        'message': 'Model yeniden yüklendi' if model is not None else 'Model yüklenemedi'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
