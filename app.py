from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply
from datetime import datetime
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ======================================================
# CONFIGURATION
# ======================================================
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ======================================================
# CHANNEL ATTENTION (for ocular model)
# ======================================================
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    d1 = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    d2 = Dense(channel, kernel_initializer='he_normal', use_bias=True)
    
    avg = GlobalAveragePooling2D()(input_feature)
    avg = Reshape((1, 1, channel))(avg)
    avg = d1(avg)
    avg = d2(avg)
    
    maxp = GlobalMaxPooling2D()(input_feature)
    maxp = Reshape((1, 1, channel))(maxp)
    maxp = d1(maxp)
    maxp = d2(maxp)
    
    attn = tf.keras.layers.Add()([avg, maxp])
    attn = tf.keras.activations.sigmoid(attn)
    
    return Multiply()([input_feature, attn])

# ======================================================
# MODEL PATHS
# ======================================================
DR_MB_PATH = "models/multibranch_model_1.h5"
DR_CNN_PATH = "models/cnn_model_1.h5"
OCULAR_PATH = "models/hybrid_efficientnetb4_model.keras"
HP_PATH = "models/final_hypertension_model.h5"

# ======================================================
# LOAD MODELS
# ======================================================
print("🔄 Loading AI models...")
MODELS_LOADED = False
dr_mb = None
dr_cnn = None
ocular_model = None
hp_model = None

try:
    dr_mb = tf.keras.models.load_model(DR_MB_PATH, compile=False)
    print("✅ Diabetic Retinopathy MultiBranch model loaded")
    
    dr_cnn = tf.keras.models.load_model(DR_CNN_PATH, compile=False)
    print("✅ Diabetic Retinopathy CNN model loaded")
    
    ocular_model = tf.keras.models.load_model(OCULAR_PATH, custom_objects={"channel_attention": channel_attention})
    print("✅ Ocular Disease model loaded")
    
    hp_model = tf.keras.models.load_model(HP_PATH, compile=False)
    print("✅ Hypertension model loaded")
    
    MODELS_LOADED = True
    print("\n✅ All models loaded successfully!\n")
except Exception as e:
    print(f"⚠️  Models not found: {str(e)}")
    print("🔄 Running in DEMO MODE with simulated predictions")
    print("📝 For production, ensure model files are in the 'models/' folder\n")
    MODELS_LOADED = False

# ======================================================
# CLASS LABELS
# ======================================================
DR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

OCULAR_CLASSES = ["Normal", "Cataract", "Glaucoma", "Retina Disease"]

# ======================================================
# PREPROCESSING FUNCTIONS
# ======================================================
def preprocess_dr(path):
    """Preprocess image for Diabetic Retinopathy models"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Failed to load image")
    
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_ocular(path):
    """Preprocess image for Ocular Disease model"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Failed to load image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (380, 380))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_hp(path):
    """Preprocess image for Hypertension model"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Failed to load image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def risk_level(p):
    """Determine risk level based on probability"""
    if p >= 0.6:
        return "HIGH"
    elif p >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"

# ======================================================
# MOCK PREDICTION FUNCTIONS (FOR DEMO MODE)
# ======================================================
def generate_mock_dr_prediction():
    """Generate realistic mock DR prediction"""
    stages = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    # Weighted random selection (more common stages have higher probability)
    weights = [0.4, 0.3, 0.2, 0.07, 0.03]
    stage_idx = np.random.choice(len(stages), p=weights)
    
    # Generate probabilities that sum to 1
    probs = np.random.dirichlet(np.ones(5) * 0.5)
    # Make the selected stage have highest probability
    probs[stage_idx] += 0.5
    probs = probs / probs.sum()
    
    return stage_idx, probs

def generate_mock_ocular_prediction():
    """Generate realistic mock ocular disease prediction"""
    # Typically one disease dominates, others are low
    dominant_idx = np.random.randint(0, 4)
    probs = np.random.uniform(0.05, 0.15, 4)
    probs[dominant_idx] = np.random.uniform(0.6, 0.9)
    probs = probs / probs.sum()
    return probs

def generate_mock_hp_prediction():
    """Generate realistic mock hypertension prediction"""
    return np.random.uniform(0.1, 0.9)

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def save_analysis_result(user_id, result_data):
    """Save analysis results to JSON file"""
    result_file = os.path.join(RESULTS_FOLDER, f'{user_id}_results.json')
    
    # Load existing results
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    # Add new result
    result_data['timestamp'] = datetime.now().isoformat()
    results.append(result_data)
    
    # Save updated results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

# ======================================================
# ROUTES
# ======================================================
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': MODELS_LOADED,
        'demo_mode': not MODELS_LOADED
    })

@app.route('/api/user/history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    """Get analysis history for a user"""
    result_file = os.path.join(RESULTS_FOLDER, f'{user_id}_results.json')
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
        return jsonify({'success': True, 'history': results})
    else:
        return jsonify({'success': True, 'history': []})

# ======================================================
# PREDICTION ENDPOINT
# ======================================================
@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for retinal analysis"""
    
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Get user ID if provided
    user_id = request.form.get('user_id', 'anonymous')
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{user_id}_{timestamp}.jpg'
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(temp_path)
        print(f"📸 Processing image: {filename}")
        
        if MODELS_LOADED:
            # REAL MODEL PREDICTIONS
            # DIABETIC RETINOPATHY PREDICTION
            print("🔬 Analyzing for Diabetic Retinopathy...")
            dr_img = preprocess_dr(temp_path)
            mb_probs = dr_mb.predict(dr_img, verbose=0)[0]
            cnn_probs = dr_cnn.predict(dr_img, verbose=0)[0]
            dr_probs = 0.7 * mb_probs + 0.3 * cnn_probs
            dr_class = int(np.argmax(dr_probs))
            dr_conf = float(dr_probs[dr_class])
            
            # OCULAR DISEASE PREDICTION
            print("👁️  Screening for Ocular Diseases...")
            ocular_img = preprocess_ocular(temp_path)
            ocular_probs = ocular_model.predict(ocular_img, verbose=0)[0]
            
            # HYPERTENSION PREDICTION
            print("❤️  Detecting Hypertensive Changes...")
            hp_img = preprocess_hp(temp_path)
            hp_prob = float(hp_model.predict(hp_img, verbose=0)[0][0])
        else:
            # DEMO MODE - MOCK PREDICTIONS
            print("🎭 Generating demo predictions (models not loaded)...")
            import time
            time.sleep(2)  # Simulate processing time
            
            # Mock DR prediction
            dr_class, dr_probs_array = generate_mock_dr_prediction()
            dr_conf = float(dr_probs_array[dr_class])
            dr_probs = dr_probs_array
            
            # Mock Ocular prediction
            ocular_probs = generate_mock_ocular_prediction()
            
            # Mock Hypertension prediction
            hp_prob = generate_mock_hp_prediction()
        
        # Process results (same for both real and mock)
        ocular_results = [
            {
                "disease": cls,
                "probability": float(prob),
                "risk": risk_level(prob)
            }
            for cls, prob in zip(OCULAR_CLASSES, ocular_probs)
        ]
        
        hp_risk = risk_level(hp_prob)
        
        # Compile results
        result = {
            'diabetic_retinopathy': {
                'stage': DR_CLASSES[dr_class],
                'confidence': dr_conf * 100,
                'class_probabilities': {
                    DR_CLASSES[i]: float(dr_probs[i] * 100)
                    for i in range(len(DR_CLASSES))
                }
            },
            'ocular_diseases': ocular_results,
            'hypertension': {
                'risk_level': hp_risk,
                'probability': hp_prob * 100
            },
            'metadata': {
                'filename': filename,
                'processed_at': datetime.now().isoformat(),
                'user_id': user_id,
                'demo_mode': not MODELS_LOADED
            }
        }
        
        # Save result to history
        save_analysis_result(user_id, result)
        
        print(f"✅ Analysis complete for {filename}\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e)
        }), 500

# ======================================================
# ERROR HANDLERS
# ======================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ======================================================
# MAIN
# ======================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 OptiSense AI - Retinal Analysis System")
    print("="*60)
    print(f"📍 Server starting at: http://localhost:5000")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📊 Results folder: {RESULTS_FOLDER}")
    print(f"🤖 Demo mode: {not MODELS_LOADED}")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')