import joblib
import numpy as np
import os
from django.shortcuts import render
from django.conf import settings


# ── Load model and scaler once at startup ─────────────────────────────────────
MODEL_PATH  = os.path.join(settings.BASE_DIR, 'ml_models', 'smartphone_addiction_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler_v1.pkl')

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODEL_LOADED = True
except Exception as e:
    model  = None
    scaler = None
    MODEL_LOADED = False
    print(f"[WARNING] Could not load ML model: {e}")


# ── Feature order must match training ────────────────────────────────────────
# Step 5 in the notebook dropped: gaming_hours, sleep_hours,
# notifications_per_day, app_opens_per_day (absorbed into engineered features).
# The model and scaler were both fitted on these 8 columns only:
FEATURE_ORDER = [
    'daily_screen_time_hours',
    'social_media_hours',
    'weekend_screen_time',
    'total_usage',
    'sleep_ratio',
    'risk_ratio',
    'habit_intensity',
    'weekend_spike',
]


def engineer_features(raw: dict) -> list:
    """Compute engineered features and return values in FEATURE_ORDER (8 features).

    Raw inputs collected from the form:
        daily_screen_time_hours, social_media_hours, gaming_hours,
        sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time
    """
    st  = raw['daily_screen_time_hours']
    sm  = raw['social_media_hours']
    gm  = raw['gaming_hours']
    sl  = raw['sleep_hours']
    nt  = raw['notifications_per_day']
    ao  = raw['app_opens_per_day']
    wk  = raw['weekend_screen_time']

    total_usage     = st + sm + gm          # absorbed gaming_hours
    sleep_ratio     = sl / 24.0             # absorbed sleep_hours
    risk_ratio      = st / (sl + 0.1)       # absorbed sleep_hours
    habit_intensity = ao * nt               # absorbed both notification cols
    weekend_spike   = wk / (st + 0.1)

    # Return ONLY the 8 selected features — raw cols that were dropped in
    # Step 5 must NOT be included here or the scaler will raise a shape error.
    return [st, sm, wk,
            total_usage, sleep_ratio, risk_ratio, habit_intensity, weekend_spike]


# ── Views ─────────────────────────────────────────────────────────────────────

def home(request):
    """Landing page with project overview."""
    return render(request, 'checker/home.html', {'model_loaded': MODEL_LOADED})


def predict(request):
    """GET: show form. POST: run prediction and display result."""
    if request.method == 'POST':
        errors = {}
        raw = {}

        fields = {
            'daily_screen_time_hours': ('Daily Screen Time',   0,  24),
            'social_media_hours':      ('Social Media Hours',  0,  24),
            'gaming_hours':            ('Gaming Hours',        0,  24),
            'sleep_hours':             ('Sleep Hours',         0,  24),
            'notifications_per_day':   ('Notifications / Day', 0, 999),
            'app_opens_per_day':       ('App Opens / Day',     0, 999),
            'weekend_screen_time':     ('Weekend Screen Time', 0,  48),
        }

        for key, (label, lo, hi) in fields.items():
            try:
                val = float(request.POST.get(key, ''))
                if not (lo <= val <= hi):
                    errors[key] = f"{label} must be between {lo} and {hi}."
                raw[key] = val
            except (ValueError, TypeError):
                errors[key] = f"{label} must be a valid number."

        if errors:
            return render(request, 'checker/predict.html', {
                'errors': errors,
                'prev': request.POST,
            })

        if not MODEL_LOADED:
            return render(request, 'checker/predict.html', {
                'model_error': True,
                'prev': request.POST,
            })

        # Engineer features and scale
        features     = np.array(engineer_features(raw)).reshape(1, -1)
        scaled       = scaler.transform(features)
        prediction   = int(model.predict(scaled)[0])
        probability  = float(model.predict_proba(scaled)[0][1])  # P(addicted)
        risk_score   = round(probability * 100, 1)

        # Risk band
        if probability >= 0.75:
            risk_band = 'high'
        elif probability >= 0.50:
            risk_band = 'moderate'
        else:
            risk_band = 'low'

        context = {
            'addicted':    prediction == 1,
            'probability': probability,
            'risk_score':  risk_score,
            'risk_band':   risk_band,
            'inputs':      raw,
            'total_usage': raw['daily_screen_time_hours'] + raw['social_media_hours'] + raw['gaming_hours'],
            'risk_ratio':  round(raw['daily_screen_time_hours'] / (raw['sleep_hours'] + 0.1), 2),
        }
        return render(request, 'checker/result.html', context)

    return render(request, 'checker/predict.html', {'prev': {}})


def about(request):
    """Project methodology and dataset info."""
    metrics = {
        'accuracy':  '91.2%',
        'precision': '93.4%',
        'recall':    '89.7%',
        'f1':        '91.5%',
        'roc_auc':   '0.964',
    }
    return render(request, 'checker/about.html', {'metrics': metrics})