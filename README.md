# 👁️ OptiSense AI - Advanced Retinal Diagnostics Platform

<div align="center">

![OptiSense AI](https://img.shields.io/badge/OptiSense-AI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-blue?style=for-the-badge&logo=react)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**A comprehensive AI-powered retinal analysis system with professional UI, authentication, and multiple disease detection capabilities.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

### 🔐 **Authentication System**
- ✅ User Registration & Login
- ✅ Secure Session Management
- ✅ Personalized Dashboard
- ✅ User Profile Management
- ✅ Logout Functionality

### 🏥 **Medical Analysis Capabilities**

#### 1️⃣ Diabetic Retinopathy Detection
- **5-Stage Classification:** No DR, Mild, Moderate, Severe, Proliferative DR
- **Ensemble Model:** Combines MultiBranch CNN + Standard CNN
- **CLAHE Preprocessing:** Enhanced image quality
- **High Accuracy:** Confidence scores for each prediction

#### 2️⃣ Ocular Disease Screening
- **4 Disease Categories:** Normal, Cataract, Glaucoma, Retina Disease
- **Risk Assessment:** HIGH, MEDIUM, LOW risk levels
- **EfficientNetB4 Architecture:** State-of-the-art deep learning
- **Probability Scores:** Detailed detection confidence

#### 3️⃣ Cardiovascular Health Indicators
- **Hypertensive Retinopathy Detection:** Vascular changes assessment
- **Clinical Findings:** Detailed medical terminology
- **Risk Stratification:** Automated risk level assignment
- **Recommendations:** Clinical guidance for each risk level

### 🎨 **Professional UI/UX**
- 🌊 **Animated Gradients:** Dynamic background effects
- ✨ **Floating Particles:** Engaging visual atmosphere
- 🔮 **Glass Morphism:** Modern frosted glass design
- 📱 **Fully Responsive:** Works on desktop, tablet, mobile
- 🌙 **Dark Theme:** Eye-friendly interface
- 🎭 **Custom Typography:** Professional serif + sans-serif pairing
- ⚡ **Smooth Animations:** Polished micro-interactions

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **pip** (Python package manager)
- **Git** ([Download](https://git-scm.com/downloads))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/optisense-ai.git
cd optisense-ai
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Model Files

Place the following AI model files in the project root:

```
optisense-ai/
├── multibranch_model_1.h5          # DR MultiBranch model
├── cnn_model_1.h5                  # DR CNN model
├── hybrid_efficientnetb4_model.keras  # Ocular disease model
└── final_hypertension_model.h5     # Hypertension model
```

### Step 4: Project Structure

```
optisense-ai/
│
├── index.html                      # Frontend application
├── app.py                          # Flask backend server
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── multibranch_model_1.h5         # AI Model
├── cnn_model_1.h5                 # AI Model
├── hybrid_efficientnetb4_model.keras  # AI Model
├── final_hypertension_model.h5    # AI Model
│
├── uploads/                        # Auto-created
└── results/                        # Auto-created
```

---

## 💻 Usage

### Starting the Server

```bash
python app.py
```

**Expected Output:**
```
============================================================
🚀 OptiSense AI - Retinal Analysis System
============================================================
📍 Server starting at: http://localhost:5000

🔄 Loading AI models...
✅ All models loaded successfully!
```

### Using the Application

1. **Open Browser:** Navigate to `http://localhost:5000`

2. **Create Account:**
   - Click "Sign up"
   - Enter name, email, password (min 6 characters)
   - Click "Sign Up"

3. **Login:**
   - Enter email and password
   - Click "Sign In"

4. **Upload & Analyze:**
   - Upload fundus image (JPG/PNG, max 10MB)
   - Click "Run Complete Analysis"
   - Wait 5-15 seconds

5. **View Results:**
   - Diabetic Retinopathy staging
   - Ocular Disease screening
   - Cardiovascular health assessment

---

## 📊 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Health Check
```http
GET /health
```

#### Predict/Analyze
```http
POST /predict
Content-Type: multipart/form-data

Body:
- image: file (JPG/PNG, max 10MB)
- user_id: string (optional)
```

**Response:**
```json
{
  "diabetic_retinopathy": {
    "stage": "Mild",
    "confidence": 87.5
  },
  "ocular_diseases": [
    {
      "disease": "Cataract",
      "probability": 0.72,
      "risk": "HIGH"
    }
  ],
  "hypertension": {
    "risk_level": "MEDIUM",
    "probability": 45.2
  }
}
```

---

## 🐛 Troubleshooting

### Common Issues

**Models Not Loading**
```
Error: Unable to open file
```
✅ Solution: Ensure all model files are in project root

**Port Already in Use**
```
Error: Address already in use
```
✅ Solution: Change port in app.py to 5001

**Image Upload Fails**
✅ Solution: Check file size (max 10MB) and format (JPG/PNG)

**Login Not Working**
✅ Solution: Ensure not in private/incognito mode (localStorage required)

---

## 🚀 Deployment

### Using Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t optisense-ai .
docker run -p 5000:5000 optisense-ai
```

---

## ⚠️ Medical Disclaimer

**IMPORTANT:** This system is a **screening and research tool** for educational purposes only.

- ✋ NOT FDA approved
- 👨‍⚕️ Requires professional medical review
- 🏥 NOT for clinical decisions
- 📋 Clinical validation required

The hypertensive retinopathy detection uses simulated data for prototype demonstration.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- TensorFlow Team
- OpenCV Community
- React Team
- Flask Team

---

## 📧 Contact

- **Issues:** [GitHub Issues](https://github.com/yourusername/optisense-ai/issues)
- **Email:** your.email@example.com

---

## 📊 Roadmap

- [x] ✅ Diabetic Retinopathy Detection
- [x] ✅ Ocular Disease Screening
- [x] ✅ User Authentication
- [x] ✅ Professional UI/UX
- [ ] 🔄 Improved Hypertension Model
- [ ] 🔄 User Dashboard with History
- [ ] 🔄 PDF Report Generation
- [ ] 🔄 Multi-language Support
- [ ] 🔄 Mobile App

---

<div align="center">

**Made with ❤️ for better eye health**

[⬆ Back to Top](#-optisense-ai---advanced-retinal-diagnostics-platform)

</div>
