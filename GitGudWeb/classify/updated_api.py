import os
import base64
import io
import uuid
import cv2
import time
import glob
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_classification import predict, visualize_masks

app = Flask(__name__)
CORS(app)

# Model ve ağırlık dosyaları
MODEL_PATH = "best_twin_vit.pth"
YOLO_WEIGHTS_PATH = "best.pt"
RESULTS_PATH = "prediction_masks.pkl"

# Veri dizinleri
DATASET_DIR = "classification"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
VISUALIZE_DIR = "visualize_preds"

# Klasörleri oluştur
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(VISUALIZE_DIR, exist_ok=True)

def base64_to_image(b64_string, image_id, is_pre=True):
    try:
        if ',' in b64_string:
            b64_string = b64_string.split(',', 1)[1]
        data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(data))
        suffix = "pre_disaster.png" if is_pre else "post_disaster.png"
        filename = f"{image_id}_{suffix}"
        path = os.path.join(IMAGES_DIR, filename)
        img.save(path)
        return path
    except Exception as e:
        print(f"Görüntü dönüştürme hatası: {e}")
        return None


def mask_to_base64(mask):
    try:
        _, buf = cv2.imencode('.png', mask)
        b64 = base64.b64encode(buf).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Maske dönüştürme hatası: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze_damage():
    try:
        start_time = time.time()
        data = request.json or {}
        pre_b64 = data.get('preImage')
        post_b64 = data.get('postImage')
        if not pre_b64 or not post_b64:
            return jsonify({"error": "Afet öncesi ve sonrası görüntüler gereklidir"}), 400

        image_id = uuid.uuid4().hex[:8]
        pre_path = base64_to_image(pre_b64, image_id, is_pre=True)
        post_path = base64_to_image(post_b64, image_id, is_pre=False)
        if not pre_path or not post_path:
            return jsonify({"error": "Görüntüler kaydedilemedi"}), 500

        print(f"Hasar analizi yapılıyor... ID: {image_id}")
        results = predict(
            base_path=DATASET_DIR,
            yolo_weights_path=YOLO_WEIGHTS_PATH,
            model_path=MODEL_PATH,
            results_path=RESULTS_PATH,
            id_list=[image_id]
        )

        # Görselleştirme yap
        try:
            visualize_masks(DATASET_DIR, results, labels=False)
            print("Görselleştirme tamamlandı")
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

        # Binaları işle
        buildings = []
        key = f"{image_id}_post_disaster.png"
        if key in results:
            for bbox, dmg, mask in results[key]:
                x1, y1, x2, y2 = map(int, bbox)
                b64_mask = mask_to_base64(mask)
                buildings.append({
                    "bbox": [x1, y1, x2, y2],
                    "damage": dmg,
                    "mask": b64_mask
                })

        # İstatistikleri hesapla
        stats = {"no-damage":0, "minor-damage":0, "major-damage":0, "destroyed":0}
        for b in buildings:
            stats[b["damage"]] += 1

        # Görselleştirilmiş maskeyi bul ve base64'e dönüştür
        masked_image = None
        try:
            # Görselleştirilmiş dosyayı bul
            pred_files = glob.glob(os.path.join(VISUALIZE_DIR, f"{image_id}_pred.png"))
            if pred_files:
                with open(pred_files[0], "rb") as img_file:
                    masked_image = base64.b64encode(img_file.read()).decode('utf-8')
                    masked_image = f"data:image/png;base64,{masked_image}"
        except Exception as e:
            print(f"Görselleştirilmiş maske dönüştürme hatası: {e}")

        # Sonuçları döndür
        response = {
            "success": True,
            "image_id": image_id,
            "buildings": buildings,
            "statistics": stats,
            "total_buildings": len(buildings)
        }
        
        # Maskelenmiş görüntü varsa ekle
        if masked_image:
            response["masked_image"] = masked_image
        
        print(f"Analiz tamamlandı. Süre: {time.time() - start_time:.2f} saniye")
        return jsonify(response)
    
    except Exception as e:
        print(f"Analiz sırasında hata: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Analiz sırasında bir hata oluştu: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    ok = os.path.exists(MODEL_PATH) and os.path.exists(YOLO_WEIGHTS_PATH)
    return jsonify({
        "status": "healthy" if ok else "unhealthy",
        "model_path": MODEL_PATH,
        "yolo_path": YOLO_WEIGHTS_PATH,
        "dataset_dir": DATASET_DIR
    })

@app.route('/cleanup', methods=['POST'])
def cleanup_images():
    try:
        for f in os.listdir(IMAGES_DIR):
            path = os.path.join(IMAGES_DIR, f)
            if os.path.isfile(path): os.remove(path)
        return jsonify({"success": True, "message": "Geçici görüntüler temizlendi"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
