from django.urls import path
from . import views

urlpatterns = [
    # This maps the root of your app to the check_addiction view
    path('', views.check_addiction, name='check_addiction'),
]