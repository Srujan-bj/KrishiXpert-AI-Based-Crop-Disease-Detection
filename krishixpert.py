from flask import Flask, request, jsonify , Response
from flask_cors import CORS
from fastai.vision.all import *
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt    
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from pymongo import MongoClient
import certifi
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import time
import psutil
import logging

# MongoDB setup
uri = "mongodb+srv://."
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client["."]
collection = db["."]

log_messages = []
# ---------------- NPK RATIOS ---------------- #

npk_ratios = {
    "Tomato__Target_Spot":                         {"N": 1, "P": 1, "K": 3},
    "Tomato__Tomato_mosaic_virus":                 {"N": 3, "P": 2, "K": 2},
    "Tomato__Tomato_YellowLeaf_Curl_Virus":        {"N": 2, "P": 3, "K": 2},
    "Tomato_Bacterial_spot":                       {"N": 2, "P": 1, "K": 3},
    "Tomato_Early_blight":                         {"N": 3, "P": 1, "K": 2},
    "Tomato_healthy":                              {"N": 1, "P": 1, "K": 1},
    "Tomato_Late_blight":                          {"N": 1, "P": 2, "K": 4},
    "Tomato_Leaf_Mold":                            {"N": 2, "P": 1, "K": 3},
    "Tomato_Septoria_leaf_spot":                   {"N": 1, "P": 1, "K": 2},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"N": 3, "P": 1, "K": 2},
}

ESP32_BASE = os.environ.get("ESP32_BASE", "http://10.101.128.151/capture")
ESP32_PUMP_URL = os.environ.get("ESP32_PUMP_URL", "http://10.101.128.56/set_npk")

MAX_TRIES = 6
RETRY_DELAY = 0.5 
APP_DEBUG = True

app = Flask(__name__)
CORS(app)

# Load FastAI model
learn = load_learner(r"C:\Users\sruja\Desktop\Plant Village\plant_disease_classifier1.pkl")

def get_green_thresholds(disease):
    disease = disease.lower()

    if "tomato_healthy" in disease:
        return np.array([35, 40, 40]), np.array([85, 255, 255])

    elif "tomato_early_blight" in disease:
        return np.array([40, 30, 50]), np.array([85, 255, 255])

    elif "tomato_late_blight" in disease:
        return np.array([28, 20, 30]), np.array([95, 255, 255])

    elif "tomato_leaf_mold" in disease:
        return np.array([28, 30, 30]), np.array([80, 255, 255])

    elif "tomato_septoria_leaf_spot" in disease:
        return np.array([32, 45, 45]), np.array([85, 255, 255])

    elif "tomato_target_spot" in disease:
        return np.array([32, 45, 45]), np.array([85, 255, 255])

    elif "tomato_bacterial_spot" in disease:
        return np.array([35, 55, 55]), np.array([85, 255, 255])

    elif "tomato_tomato_yellowleaf_curl_virus" in disease:
        return np.array([40, 75, 75]), np.array([85, 255, 255])

    elif "tomato_tomato_mosaic_virus" in disease:
        return np.array([35, 50, 50]), np.array([85, 255, 255])

    elif "tomato_spider_mites_two_spotted_spider_mite" in disease:
        return np.array([38, 60, 60]), np.array([85, 255, 255])

    else:
        return np.array([35, 40, 40]), np.array([85, 255, 255])

def generate_analysis_images(image, disease_name):

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask2 = isolate_leaf(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = get_green_thresholds(disease_name)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    healthy_leaf = cv2.bitwise_and(green_mask, green_mask, mask=mask2)

    total_foreground_pixels = np.count_nonzero(mask2)
    healthy_pixels = np.count_nonzero(healthy_leaf)

    healthy_percentage = (healthy_pixels / total_foreground_pixels) * 100
    unhealthy_percentage = 100 - healthy_percentage

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img1 = cv2.bitwise_and(image, image, mask=mask2)
#healthy in green
    green_white = np.full_like(img1, (255, 255, 255))

    green_white[(green_mask == 255) & (mask2 == 1)] = (0, 255, 0)

    cv2.drawContours(green_white, contours, -1, (0, 0, 0), 2)
#unhealthy in red
    white_red = np.full_like(img1, (255, 255, 255))
    white_red[(green_mask == 0) & (mask2 == 1)] = (0, 0, 255)
    cv2.drawContours(white_red, contours, -1, (0, 0, 0), 2)

    def to_base64(img_bgr):
        _, buffer = cv2.imencode('.png', img_bgr)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

    def histogram_base64(img_bgr):
        plt.figure()
        for i, col in enumerate(('b', 'g', 'r')):
            hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.ylim(0, 5000)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    return {
        'original_black_bg': to_base64(img1),
        'healthy_green_unhealthy_white': to_base64(green_white),
        'healthy_white_unhealthy_red': to_base64(white_red),
        'color_histogram1': histogram_base64(img1),
        'color_histogram2': histogram_base64(green_white),
        'color_histogram3': histogram_base64(white_red),
        'healthy_percent': round(healthy_percentage, 2),
        'unhealthy_percent': round(unhealthy_percentage, 2)
    }


def isolate_leaf(cv_image):
    mask = np.zeros(cv_image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, cv_image.shape[1] - 20, cv_image.shape[0] - 20)
    cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, 20, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

def confidence_graph(x, y):
    plt.figure()
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('Confidence vs Loss')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Loss (%)')
    plt.ylim(-10, 110)
    plt.xlim(-10, 110)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def hnun_graph(x, y):
    plt.figure()
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('Healthy vs Unhealthy')
    plt.xlabel('Healthy (%)')
    plt.ylabel('Unhealthy (%)')
    plt.ylim(-10, 110)
    plt.xlim(-10, 110)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


#PUMP LOGIC

def calculate_pump_times(disease: str,
                         unhealthy_percent: float,
                         total_spray_ms: int = 10000):

    ratio = npk_ratios.get(disease, {"N": 1, "P": 1, "K": 1})
    rN, rP, rK = ratio["N"], ratio["P"], ratio["K"]
    s = rN + rP + rK if (rN + rP + rK) != 0 else 1

    baseN = total_spray_ms * (rN / s)
    baseP = total_spray_ms * (rP / s)
    baseK = total_spray_ms * (rK / s)


    up = float(unhealthy_percent)
    scale = max(0.0, min(up, 100.0)) / 100.0

    tN = int(baseN * scale)
    tP = int(baseP * scale)
    tK = int(baseK * scale)
    tS = int(total_spray_ms * scale * 2)

    return {
        "pumpN_ms": tN,
        "pumpP_ms": tP,
        "pumpK_ms": tK,
        "pumpS_ms": tS
    }


def send_to_esp32_pump(disease: str,
                       unhealthy_percent: float,
                       pump_times: dict):

    if not ESP32_PUMP_URL:
        return None, "ESP32_PUMP_URL not configured"

    payload = {
        "disease": disease,
        "unhealthy_percentage": unhealthy_percent,
        "pump_times": pump_times
    }  
    try:
        resp = requests.post(ESP32_PUMP_URL, json=payload, timeout=5)
        return resp.status_code, resp.text
    except Exception as e:
        return None, f"Error sending to ESP32 pump: {e}"



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/system_stats")
def system_stats():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.5)

    return {
        "cpu_percent": cpu,
        "ram_used_mb": round(memory.used / (1024 * 1024), 2),
        "ram_total_mb": round(memory.total / (1024 * 1024), 2),
        "ram_percent": memory.percent
    }

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.perf_counter()
    
    final_image_bytes = None
    if final_image_bytes is None:
        previous_hash = None
        attempts = 0

        for attempt in range(1, MAX_TRIES + 1):
                attempts = attempt
                ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
                unique = f"{ts}_{random.randint(1000, 9999)}"
                esp_url = f"{ESP32_BASE}?t={unique}"

                try:
                    resp = requests.get(
                        esp_url,
                        headers={
                            "Cache-Control": "no-cache, no-store, must-revalidate",
                            "Pragma": "no-cache",
                            "Connection": "close",
                        },
                        timeout=15,
                        stream=False
                    )
                except requests.RequestException as e:
                    app.logger.warning("ESP request failed (attempt %d): %s", attempt, e)
                    time.sleep(RETRY_DELAY)
                    continue

                if resp.status_code != 200:
                    app.logger.warning("ESP returned non-200 (attempt %d): %s", attempt, resp.status_code)
                    resp.close()
                    time.sleep(RETRY_DELAY)
                    continue

                image_bytes = resp.content
                resp.close()

                if not image_bytes or len(image_bytes) < 500:
                    app.logger.warning("Captured image too small/empty (attempt %d).", attempt)
                    time.sleep(RETRY_DELAY)
                    continue

                cur_hash = hashlib.md5(image_bytes).hexdigest()
                app.logger.debug("Attempt %d: size=%d md5=%s", attempt, len(image_bytes), cur_hash)

                if previous_hash is None:
                    previous_hash = cur_hash
                    final_image_bytes = image_bytes
                    app.logger.info("Captured initial frame (attempt %d). Trying once more...", attempt)
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    if cur_hash != previous_hash:
                        final_image_bytes = image_bytes
                        app.logger.info("Captured fresh frame on attempt %d (md5 changed)", attempt)
                        break
                    else:
                        previous_hash = cur_hash
                        app.logger.info("Attempt %d returned identical frame; retrying...", attempt)
                        time.sleep(RETRY_DELAY)
                        continue

        if final_image_bytes is None:
                return jsonify({"error": "Failed to obtain image (no uploaded file and ESP32 capture failed)."}), 500

        app.logger.info("Proceeding with captured image (attempts used: %d)", attempts)
    else:
            attempts = 0


    img = Image.open(BytesIO(final_image_bytes)).convert("RGB")
    npk = (0,0,0)  # Default value
    pred, pred_idx, probs = learn.predict(img)
    confidence = round(float(probs[pred_idx]) * 100, 2)
    loss = round(100 - confidence, 2)
   

    
    analysis_data = generate_analysis_images(img,pred)
    healthy_percentage = analysis_data['healthy_percent']
    unhealthy_percentage = analysis_data['unhealthy_percent']
    timestamp = datetime.now().isoformat()
    timestamp1 = datetime.now()
    # Store prediction and analysis in MongoDB
    collection.insert_one({
        "PREDICTION": pred,
        "npk":npk,
        "ACCURACY": confidence,
        "LOSS": loss,
        "HEALTHY PERCENTAGE": healthy_percentage,
        "UNHEALTHY PERCENTAGE": unhealthy_percentage,
        "TIMESTAMP": timestamp,
        "image_black_background": analysis_data['original_black_bg'],
        "image_green_white": analysis_data['healthy_green_unhealthy_white'],
        "image_white_red": analysis_data['healthy_white_unhealthy_red'],
        "histogram1": analysis_data['color_histogram1'],
        "histogram2": analysis_data['color_histogram2'],
        "histogram3": analysis_data['color_histogram3'],
        "probs": probs.tolist()  # Store per-class probabilities for ROC
    })

    # Prepare data for graphs and metrics
    x_conf, y_loss, x_healthy, y_unhealthy = [], [], [], []
    y_true, y_pred, y_probs = [], [], []

    for doc in collection.find({}, {"_id": 0}):
        if 'PREDICTION' in doc:
            y_true.append(doc['PREDICTION'])
            y_pred.append(doc['PREDICTION'])
            x_conf.append(doc.get('ACCURACY', 0))
            y_loss.append(doc.get('LOSS', 0))
            x_healthy.append(doc.get('HEALTHY PERCENTAGE', 0))
            y_unhealthy.append(doc.get('UNHEALTHY PERCENTAGE', 0))
            if 'probs' in doc:
                y_probs.append(doc['probs'])

    # Encode confusion matrix
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = le.transform(y_pred)
    n_classes = len(le.classes_)

    cm = confusion_matrix(y_true_enc, y_pred_enc)

    plt.figure(figsize=(10, 8))   

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        square=True,
        cbar=True,
        annot_kws={"size": 10}
                )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)

    plt.xticks(rotation=45, ha='right')  
    plt.yticks(rotation=0)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)  
    plt.close()
    buf.seek(0)

    cm_img = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    # ROC Curve (multiclass)
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true_enc, classes=range(n_classes))
    y_probs_arr = np.array(y_probs)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs_arr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_probs_arr.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))

    plt.plot(fpr["micro"], tpr["micro"],
         label=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
         linewidth=3)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.title("ROC Curve", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)

    roc_img = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    cont = collection.count_documents({})
    end_time = time.perf_counter()
    prediction_latency = round((end_time - start_time) * 1000, 2)
    pred_str = str(pred)
    npk_ratio = npk_ratios.get(pred_str, {"N": 1, "P": 1, "K": 1})
    pump_times = calculate_pump_times(pred_str, unhealthy_percentage)
    esp32_pump_status, esp32_pump_reply = send_to_esp32_pump(pred_str, unhealthy_percentage, pump_times)


    return jsonify({
        'prediction': pred,
        'loss': loss,
        'confidence': confidence,
        'cause': ' ',
        'remedies': ' ',
        'count': cont,
        'confidence_graph': confidence_graph(x_conf, y_loss),
        'healthy_percentage': healthy_percentage,
        'unhealthy_percentage': unhealthy_percentage,
        'handuh_graph': hnun_graph(x_healthy, y_unhealthy),
        'image_black_background': analysis_data['original_black_bg'],
        'image_green_white': analysis_data['healthy_green_unhealthy_white'],
        'image_white_red': analysis_data['healthy_white_unhealthy_red'],
        'histogram1': analysis_data['color_histogram1'],
        'histogram2': analysis_data['color_histogram2'],
        'histogram3': analysis_data['color_histogram3'],
        'timestamp': timestamp1,
        'confusion_matrix': cm_img,
        'roc_curve': roc_img,
        'prediction_latency': prediction_latency,
        'npk_ratio': npk_ratio,
        'pump_times_ms': pump_times,
        'esp32_pump_status': esp32_pump_status,
        'esp32_pump_reply': esp32_pump_reply,


    })

if __name__ == '__main__':
    app.run(debug=True ,threaded=True)

