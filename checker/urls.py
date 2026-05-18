from django.urls import path
from . import views

app_name = 'checker'

urlpatterns = [
    path('',        views.home,    name='home'),
    path('predict/', views.predict, name='predict'),
    path('about/',   views.about,   name='about'),
]