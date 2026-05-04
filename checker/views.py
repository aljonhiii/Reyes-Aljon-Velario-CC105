import os
import pandas as pd
import joblib
from django.shortcuts import render
from .forms import AddictionForm
from django.conf import settings

# 1. LOAD MODELS SAFELY (Once when the server starts)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'checker/ml_models/smartphone_addiction_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'checker/ml_models/scaler_v1.pkl')

# Global variables for the model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Initial verification print
if hasattr(model, 'n_estimators'):
    print(f"SUCCESS: Loaded Random Forest with {model.n_estimators} trees.")
else:
    print("WARNING: This object does not look like your trained Random Forest.")

def check_addiction(request):
    result = None
    css_class = ""
    raw_prediction = None 

    if request.method == 'POST':
        form = AddictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            
            # 2. CREATE THE DATAFRAME
            df = pd.DataFrame([{
                'daily_screen_time_hours': data['daily_screen_time'],
                'social_media_hours': data['social_media'],
                'gaming_hours': data['gaming'],
                'work_study_hours': 0, 
                'sleep_hours': data['sleep_hours'],
                'notifications_per_day': data['notifications'],
                'app_opens_per_day': data['app_opens'],
                'weekend_screen_time': data['weekend_usage'],
                'stress_level': 0,           
                'academic_work_impact': 0    
            }])

            # 3. FEATURE ENGINEERING
            df['total_usage'] = df['daily_screen_time_hours'] + df['social_media_hours'] + df['gaming_hours']
            df['sleep_ratio'] = df['sleep_hours'] / 24.0
            df['risk_ratio'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 0.1)
            df['habit_intensity'] = df['app_opens_per_day'] * df['notifications_per_day']
            df['weekend_spike'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + 0.1)
            
            # 4. PREPARE DATA FOR MODEL
            # Ensure columns match training order and apply scaling
            df = df[scaler.feature_names_in_]
            scaled_input = scaler.transform(df)

            # 5. MAKE PREDICTIONS
            raw_prediction = model.predict(scaled_input)[0] 
            prob_addicted = model.predict_proba(scaled_input)[0][1] 

            # --- DEBUG PRINTS (Moved here so variables are defined) ---
            print(f"--- NEW PREDICTION ATTEMPT ---")
            print(f"DEBUG: Model Type: {type(model)}")
            print(f"Input Data (Scaled): {scaled_input}")
            print(f"Model Prediction (Raw): {raw_prediction}")

            # 6. RISK LEVEL LOGIC
            if prob_addicted < 0.35:
                result = "✅ Low Risk / Healthy"
                css_class = "low-risk"
            elif 0.35 <= prob_addicted < 0.60:
                result = "⚠️ Mild Addiction Risk"
                css_class = "mild-risk"
            elif 0.60 <= prob_addicted < 0.85:
                result = "🚨 Severe Addiction Risk"
                css_class = "severe-risk"
            else:
                result = "⛔ High Addiction / Critical"
                css_class = "high-risk"
            
    else:
        form = AddictionForm()
        
    return render(request, 'checker/index.html', {
        'form': form, 
        'result': result, 
        'css_class': css_class,
        'raw_prediction': raw_prediction 
    })