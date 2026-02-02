from flask import Flask, request, jsonify
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize

# MongoDB setup
uri = "YOUR ULR "
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client["BASE NAME"]
collection = db["COLLECTION NAME"]

app = Flask(__name__)
CORS(app)

# Load FastAI model
learn = load_learner(r"KrishiXpert\Ai-model\plant_disease_classifier1.pkl")



def generate_analysis_images(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask2 = isolate_leaf(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)

    total_foreground_pixels = np.count_nonzero(mask2)
    healthy_percentage = (np.count_nonzero(green_mask) / total_foreground_pixels) * 100
    unhealthy_percentage = 100 - healthy_percentage

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Image 1: Original with black background
    img1 = cv2.bitwise_and(image, image, mask=mask2)

    # Image 2: Healthy in green, unhealthy in white, outline
    green_white = np.zeros_like(image)
    green_white[green_mask == 255] = (0, 255, 0)
    green_white[non_green_mask == 255] = (255, 255, 255)
    cv2.drawContours(green_white, contours, -1, (0, 0, 0), 2)

    # Image 3: Healthy in white, unhealthy in red, background white
    white_red = np.full_like(image, (255, 255, 255))
    white_red[green_mask == 255] = (255, 255, 255)
    white_red[non_green_mask == 255] = (0, 0, 255)
    white_red[mask2 == 0] = (255, 255, 255)
    cv2.drawContours(white_red, contours, -1, (0, 0, 0), 2)

    def to_base64(img_bgr):
        _, buffer = cv2.imencode('.png', img_bgr)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

    def histogram_base64(img_bgr):
        plt.figure()
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title('Color Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.ylim(0, 1000)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    img_bytes = file.read()
    img = PILImage.create(BytesIO(img_bytes)).resize((224, 224))
    npk = (0,0,0)  # Default value
    pred, pred_idx, probs = learn.predict(img)
    confidence = round(float(probs[pred_idx]) * 100, 2)
    loss = round(100 - confidence, 2)
   #more disease-specific assignments here as needed

    
    analysis_data = generate_analysis_images(img)
    healthy_percentage = analysis_data['healthy_percent']
    unhealthy_percentage = analysis_data['unhealthy_percent']
    timestamp = datetime.now().isoformat()
    
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    cm_img = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    # ROC Curve (multiclass)
    roc_img = None
    if n_classes > 2 and y_probs and np.array(y_probs).shape[1] == n_classes:
        y_true_bin = label_binarize(y_true_enc, classes=range(n_classes))
        y_probs_arr = np.array(y_probs)
        plt.figure()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_arr[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {le.classes_[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.title('Multiclass ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        roc_img = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    elif n_classes == 2 and x_conf:
        fpr, tpr, _ = roc_curve(y_true_enc, x_conf)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        roc_img = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    cont = collection.count_documents({})

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
        'timestamp': timestamp,
        'confusion_matrix': cm_img,
        'roc_curve': roc_img
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
