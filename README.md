# KrishiXpert-AI-Based-Crop-Disease-Detection
It is an AI-based agriculture support system designed to help farmers detect tomato leaf diseases using image classification. The platform provides disease identification, healthy vs unhealthy leaf analysis, remedies, and smart farming assistance through a user-friendly web interface.It includes color histogram visualization, healthy vs unhealthy leaf analysis, and ROC curve evaluation to measure model performance.

# Features

- **Tomato Leaf Disease Detection** using AI
- **Healthy vs Unhealthy Leaf Area Calculation**
- **Farmer Support Chatbot**
- **Agriculture Product Suggestions**
- **color histogram visualization**
- **confusion matrix analysis**
- **ROC curve evaluation**
- **User-Friendly Web Interface**

---

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **AI/ML:** FastAI, CNN Model  
- **Database:** MongoDB  
- **Image Processing:** OpenCV  
- **Deployment Ready**

---

## System Architecture

1. User uploads leaf image
2. Image preprocessing & background removal
3. AI model predicts disease
4. Health percentage calculation
5. Remedies & suggestions displayed

---

## Supported Diseases

- Tomato Target Spot  
- Tomato Mosaic Virus  
- Tomato Yellow Leaf Curl Virus  
- Bacterial Spot  
- Early Blight  
- Healthy Leaf  
- Late Blight  
- Leaf Mold  
- Septoria Leaf Spot  
- Spider Mites (Two-Spotted Spider Mite)  

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/KrishiXpert-AI-Agriculture-Platform.git
cd KrishiXpert
pip install -r backend/requirements.txt
python backend/app.py
