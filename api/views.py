import os
import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from datetime import datetime

# 모델 로딩
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'personal_color_model.h5'))
model = tf.keras.models.load_model(MODEL_PATH)

# 클래스 이름 불러오기
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'class_names.txt')
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f]

class PredictColorView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file = request.FILES.get('image')
        if not file:
            return Response({"error": "이미지 파일이 없습니다."}, status=400)

        # 저장 경로 생성 및 고유 파일명 생성
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.name}"
        file_path = os.path.join(settings.MEDIA_ROOT, filename)

        # 파일 저장
        try:
            with open(file_path, 'wb+') as f:
                for chunk in file.chunks():
                    f.write(chunk)
        except Exception as e:
            return Response({"error": f"파일 저장 중 오류: {str(e)}"}, status=500)

        # 이미지 로딩
        img = cv2.imread(file_path)
        if img is None:
            return Response({"error": "이미지를 읽을 수 없습니다."}, status=400)

        # 전처리
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # 예측
        try:
            pred = model.predict(img)
            predicted_class = class_names[np.argmax(pred)]
            confidence = float(np.max(pred) * 100)
        except Exception as e:
            return Response({"error": f"모델 예측 오류: {str(e)}"}, status=500)

        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        # 결과 반환
        return Response({
            "result": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })
