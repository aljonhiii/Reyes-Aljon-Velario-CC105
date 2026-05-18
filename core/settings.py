# checker/settings.py
from pathlib import Path
import os
import dj_database_url # Needed if you switch to PostgreSQL

BASE_DIR = Path(__file__).resolve().parent.parent

# Optional: Load env variables locally if you are using python-dotenv
# from dotenv import load_dotenv
# load_dotenv()


# SECURITY WARNING: keep the secret key used in production secret!
# -------------------------------------------------------------
# Fetch SECRET_KEY from the environment variable. It must be unique and strong.
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
# -------------------------------------------------------------
# Default to False if the variable is not set.
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Must define Allowed Hosts for production
# -------------------------------------------------------------
# Replace 'your-app-name' with your actual Render service name.
# Include 'localhost' for local development.
ALLOWED_HOSTS = ['your-app-name.onrender.com', 'localhost', '127.0.0.1']


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    'checker',
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    'whitenoise.middleware.WhiteNoiseMiddleware', # Correct placement
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "core.wsgi.application"


# Database
# -------------------------------------------------------------
# If deploying to Render with a persistent database, you should use PostgreSQL.
# The `dj_database_url` library handles parsing the `DATABASE_URL` env var from Render.

DATABASES = {
    'default': dj_database_url.config(
        # The `DATABASE_URL` environment variable will be provided by Render's PostgreSQL database.
        default=f'sqlite:///{BASE_DIR / "db.sqlite3"}',
        conn_max_age=600, # Recommended connection pooling setting
    )
}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# -------------------------------------------------------------
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# WhiteNoise is good for serving static files on Render.
# Ensure you are running `python manage.py collectstatic` during your deployment.
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'


# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"