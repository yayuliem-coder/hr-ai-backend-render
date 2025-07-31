from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Nama file .pkl yang baru setelah update
MODEL_FILE = 'model_performa_new.pkl'
SCALER_FILE = 'scaler_new.pkl'
LABEL_ENCODER_FILE = 'label_encoder_new.pkl'

# Memuat model, scaler, dan label encoder saat aplikasi dimulai
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    print(f"Model ({MODEL_FILE}), Scaler ({SCALER_FILE}), dan Label Encoder ({LABEL_ENCODER_FILE}) berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: Salah satu file model tidak ditemukan. Pastikan '{e.filename}' ada di folder root.")
    model = None
    scaler = None
    label_encoder = None
except Exception as e:
    print(f"Error memuat file model atau komponen lainnya: {e}")
    model = None
    scaler = None
    label_encoder = None

# Daftar kolom fitur sesuai urutan saat training model di Colab dengan data CSV baru Anda
# PASTIKAN URUTAN INI SAMA PERSIS DENGAN 'FEATURES_ORDER' DI COLAB ANDA!
FEATURES_ORDER = [
    'Tangram_Accuracy (%)', 'Tangram_Completion_Time (s)', 'Tangram_Difficulty_Level',
    'GSR_Baseline (μS)', 'GSR_Challenge (μS)', 'Pupil_Baseline (mm)',
    'Pupil_Task (mm)', 'Cognitive_Load_Index', 'Stress_Reactivity_Index'
]

@app.route('/')
def home():
    return "API Prediksi Performa HR AI berjalan di Render.com!"

@app.route('/predict_performance', methods=['POST'])
def predict_performance():
    if not model or not scaler or not label_encoder:
        return jsonify({'error': 'Model atau komponen lainnya gagal dimuat. Mohon periksa log server.'}), 500

    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # Validasi dan urutan fitur
        if not all(feature in input_df.columns for feature in FEATURES_ORDER):
            missing_features = [f for f in FEATURES_ORDER if f not in input_df.columns]
            return jsonify({'error': f"Data input tidak lengkap. Fitur yang hilang: {missing_features}. Pastikan semua kolom yang dibutuhkan ada dan namanya benar, serta sesuai urutan: {FEATURES_ORDER}"}), 400

        input_df = input_df[FEATURES_ORDER]

        processed_data = scaler.transform(input_df)

        prediction_encoded = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)

        predicted_label = label_encoder.inverse_transform(prediction_encoded)[0]

        proba_dict = {
            label_encoder.classes_[i]: round(float(prediction_proba[0][i]), 4)
            for i in range(len(label_encoder.classes_))
        }

        return jsonify({
            'predicted_performance': predicted_label,
            'prediction_probabilities': proba_dict
        })

    except KeyError as ke:
        return jsonify({'error': f"Data input tidak lengkap atau format kolom salah: {ke}. Pastikan semua kolom yang dibutuhkan ada dan namanya benar."}), 400
    except Exception as e:
        print(f"Error during prediction: {e}") 
        return jsonify({'error': str(e)}), 400

# Render.com secara otomatis akan mencari file WSGI seperti 'wsgi.py' atau 'app.py' atau 'main.py'
# dan akan menjalankan aplikasi Flask Anda.
# Tidak perlu app.run(host='0.0.0.0', ...) di Render, mereka yang mengelola servernya.
# Namun, untuk testing lokal jika Anda mau, Anda bisa tetap sertakan.
# Untuk deployment di Render, pastikan ENTRYPOINT di Render mengarah ke `main:app`
