# KrishiXpert-AI-Based-Crop-Disease-Detection
It is an AI-based agriculture support system designed to help farmers detect tomato leaf diseases using image classification, analyze NPK nutrient balance, and recommend disease-specific pesticide dosages and nutrient corrections. Instead of uniform spraying, the system enables targeted, optimized pesticide application based on infection level, reducing chemical overuse, lowering costs, protecting soil health, and improving crop yield for farmers. The platform provides disease identification, healthy vs. unhealthy leaf analysis, remedies, and smart farming assistance through a user-friendly web interface. It includes color histogram visualization, healthy vs. unhealthy leaf analysis, and ROC curve evaluation to measure model performance.

# Features

- **Tomato Leaf Disease Detection using AI**
- **IoT-Ready Smart Spraying Integration**
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

1. Autonomous Robot Moves Through Field
2. Onboard Camera Captures Leaf Images
3. Images Sent to AI Processing Unit
4. Image preprocessing & background removal
5. AI model predicts disease
6. Health percentage calculation
7. Remedies & suggestions displayed
8. Targeted Adaptive Spraying Activated

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
