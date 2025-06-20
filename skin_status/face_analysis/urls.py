# face_analysis/urls.py
from django.urls import path
from .views import analyze_faces

urlpatterns = [
    path('analyze/', analyze_faces, name='analyze-faces'),
]
