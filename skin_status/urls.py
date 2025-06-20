from django.urls import path
from .views import analyze_faces

urlpatterns = [
    path('faces/', analyze_faces, name='analyze-faces'),  # /analyze/faces/로 연결됨
]
