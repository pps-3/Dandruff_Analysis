"""
DANDRUFF ANALYSIS WEB APPLICATION
Flask-based frontend for dandruff severity prediction
"""

import os
from datetime import timedelta

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import shap


# Import core classes
from main import DandruffDataset, FeatureEngineer, MLClassifier, ClusterAnalyzer, RecommendationEngine
# Import auth blueprint
from auth import auth, login_required
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'scalpiq-secret-change-in-production')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# ── Register auth blueprint ───────────────────────────────────────
app.register_blueprint(auth)

# Global variables for models
MODEL_PATH = 'saved_models'
rf_model = None
scaler = None
clusterer = None
feature_names = None

def load_models():
    """Load trained models at startup"""
    global rf_model, scaler, clusterer, feature_names
    
    try:
        rf_model = joblib.load(os.path.join(MODEL_PATH, 'random_forest_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        
        # Load feature names
        feature_names = ['HairCareIndex', 'DietQualityScore', 'StressProxy', 
                        'SleepScore', 'age', 'gender_encoded']
        
        print("✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")
        print("   Run train_models() first to create models")
        return False

def train_models_if_needed():
    """Train models if they don't exist"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, 'random_forest_model.pkl')):
        print("\n" + "="*80)
        print("🚀 FIRST TIME SETUP - TRAINING MODELS WITH COMPARISON")
        print("="*80)
        print("\nThis will train 7 different models and select the best one.")
        print("This may take a few minutes. Please wait...")
        print("\nYou can also run 'python train_models.py' separately for detailed analysis.\n")
        
        try:
            from train_models import run_complete_training
            results = run_complete_training()
            if results:
                print("\n✓ Models trained successfully!")
                print(f"✓ Best model: {results['best_model']}")
                return True
            else:
                print("❌ Training failed!")
                return False
        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("\nFalling back to simple Random Forest training...")
            
            dataset = DandruffDataset('Dandruff_odatatset - Form Responses 1 (1).csv')
            df = dataset.load_data()
            if df is None:
                print("❌ Cannot find dataset file!")
                return False
            
            dataset.clean_column_names()
            dataset.detect_target()
            dataset.handle_missing_values()
            
            engineer = FeatureEngineer(df)
            df = engineer.engineer_all_features()
            
            feature_cols = ['HairCareIndex', 'DietQualityScore', 'StressProxy',
                           'SleepScore', 'age', 'gender_encoded']
            
            X = df[feature_cols].values
            y = df[dataset.target_column].values
            y_binned = pd.cut(y, bins=[0, 2, 3, 6], labels=['Low', 'Medium', 'High'])
            
            classifier = MLClassifier()
            classifier.prepare_data(X, y_binned, feature_cols)
            classifier.train_random_forest(use_tuning=False)
            classifier.save_model(path=MODEL_PATH)
            print("✓ Random Forest model trained and saved")
            return True
    
    return True

# ── Load models at module level so they load regardless of how Flask starts ──
# Fires whether you use `flask run`, gunicorn, or `python app.py`
train_models_if_needed()
_models_loaded = load_models()
if not _models_loaded:
    print("⚠️  WARNING: Models still not loaded after training attempt.")

def create_feature_importance_chart(feature_importance_dict):
    """Create feature importance bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(feature_importance_dict.keys())
    importances = list(feature_importance_dict.values())
    
    colors = sns.color_palette("viridis", len(features))
    ax.barh(features, importances, color=colors, alpha=0.8)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance in Your Analysis', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, v in enumerate(importances):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode()
    plt.close()
    return img

def create_severity_gauge(severity, confidence):
    """Create severity level gauge visualization"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    colors_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    
    ax.barh(['Severity'], [3], color='lightgray', height=0.3)
    ax.barh(['Severity'], [severity_map[severity] + 1],
            color=colors_map[severity], height=0.3, alpha=0.8)
    
    ax.set_xlim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_title(f'Predicted Severity: {severity} (Confidence: {confidence:.1%})',
                fontsize=14, fontweight='bold')
    ax.set_yticks([])
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode()
    plt.close()
    return img

def create_shap_chart(shap_values, feature_values_array, feature_names, predicted_class_idx):
    """Create SHAP waterfall chart for the predicted class"""
    shap_vals = shap_values[predicted_class_idx][0]

    sorted_idx = np.argsort(np.abs(shap_vals))
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = shap_vals[sorted_idx]

    n = len(sorted_features)
    bar_height = 0.4
    fig_h = max(3, n * 0.6)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    y_pos = np.arange(n)
    colors = ['#d73027' if v > 0 else '#4575b4' for v in sorted_shap]
    ax.barh(y_pos, sorted_shap, height=bar_height, color=colors, alpha=0.88)

    ax.axvline(0, color='#aaaaaa', linewidth=0.9, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=11)
    ax.set_title('SHAP Explanation — Why This Prediction?', fontsize=13, fontweight='bold', pad=14)

    x_range = sorted_shap.max() - sorted_shap.min() if sorted_shap.max() != sorted_shap.min() else 1
    pad = x_range * 0.03
    for i, v in enumerate(sorted_shap):
        ha = 'left' if v >= 0 else 'right'
        offset = pad if v >= 0 else -pad
        ax.text(v + offset, y_pos[i], f'{v:+.3f}', va='center', ha=ha, fontsize=9.5, color='#333333')

    ax.set_xlim(
        sorted_shap.min() - x_range * 0.18,
        sorted_shap.max() + x_range * 0.18
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode()
    plt.close()
    return img

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    """Main page — requires login"""
    return render_template('index.html',
                           user_name=session.get('user_name', 'User'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle prediction request"""
    try:
        # Guard: ensure models are loaded
        if scaler is None or rf_model is None:
            load_models()
        if scaler is None or rf_model is None:
            return jsonify({'success': False, 'error': 'Models are not loaded. Please check saved_models/ folder and restart the server.'}), 500

        data = request.get_json()
        
        age = int(data.get('age', 25))
        gender = data.get('gender', 'Male')
        oil_frequency = data.get('oil_frequency', 'occasionally')
        conditioner_use = data.get('conditioner_use', 'sometimes')
        water_intake = data.get('water_intake', '1-2')
        diet_type = data.get('diet_type', 'balanced')
        stress_level = data.get('stress_level', 'moderate')
        sleep_hours = data.get('sleep_hours', '7-8')
        
        oil_score = {'regularly': 1.0, 'occasionally': 0.5, 'rarely': 0.25, 'never': 0.0}.get(oil_frequency, 0.5)
        conditioner_score = {'always': 1.0, 'sometimes': 0.5, 'rarely': 0.3, 'never': 0.0}.get(conditioner_use, 0.5)
        hair_care_index = (oil_score + conditioner_score) / 2
        
        water_score = {'less_than_1': 0.2, '1-2': 0.5, '2-3': 0.8, 'more_than_3': 1.0}.get(water_intake, 0.5)
        diet_score = {'balanced': 1.0, 'irregular': 0.4, 'oily': 0.3, 'processed': 0.1}.get(diet_type, 0.4)
        diet_quality = (water_score + diet_score) / 2
        
        stress_proxy = {'high': 1.0, 'moderate': 0.6, 'minimal': 0.3, 'none': 0.0}.get(stress_level, 0.5)
        sleep_score = {'more_than_8': 1.0, '7-8': 0.9, '5-7': 0.6, 'less_than_5': 0.3}.get(sleep_hours, 0.6)
        gender_encoded = 1 if gender == 'Male' else 0
        
        features = np.array([[hair_care_index, diet_quality, stress_proxy, sleep_score, age, gender_encoded]])
        features_scaled = scaler.transform(features)
        
        prediction = rf_model.predict(features_scaled)[0]
        probabilities = rf_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        # ── SHAP explanation ──────────────────────────────────────────────────
        explainer = shap.TreeExplainer(rf_model)
        shap_raw = explainer.shap_values(features_scaled)

        if isinstance(shap_raw, np.ndarray):
            if shap_raw.ndim == 3:
                shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
            else:
                shap_values = [shap_raw]
        else:
            shap_values = shap_raw

        class_labels = list(rf_model.classes_)
        predicted_class_idx = class_labels.index(prediction)
        predicted_class_idx = min(predicted_class_idx, len(shap_values) - 1)

        shap_vals_for_class = shap_values[predicted_class_idx][0]
        shap_dict = {name: float(val) for name, val in zip(feature_names, shap_vals_for_class)}
        shap_chart = create_shap_chart(shap_values, features_scaled, feature_names, predicted_class_idx)
        # ─────────────────────────────────────────────────────────────────────
        
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        rec_engine = RecommendationEngine()
        cluster_id = 1
        recommendations = rec_engine.generate_recommendations(prediction, cluster_id)
        
        importance_chart = create_feature_importance_chart(sorted_importance)
        severity_gauge = create_severity_gauge(prediction, confidence)
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'feature_values': {
                'Hair Care Index': float(hair_care_index),
                'Diet Quality Score': float(diet_quality),
                'Stress Proxy': float(stress_proxy),
                'Sleep Score': float(sleep_score),
                'Age': age,
                'Gender': gender
            },
            'feature_importance': sorted_importance,
            'shap_values': shap_dict,
            'recommendations': recommendations,
            'charts': {
                'importance': importance_chart,
                'severity': severity_gauge,
                'shap': shap_chart
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    train_models_if_needed()
    
    if load_models():
        print("\n" + "="*70)
        print("🌐 DANDRUFF ANALYSIS WEB APP")
        print("="*70)
        print("✓ Server starting...")
        print("✓ Open your browser and go to: http://localhost:5000")
        print("="*70 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please check your dataset file.")