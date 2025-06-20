from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .utils import is_face_present, is_blurry, is_frontal_face

import numpy as np
import cv2
import os
from datetime import datetime

@api_view(['POST'])
@parser_classes([MultiPartParser])
def analyze_faces(request):
    print("✅ analyze_faces() 함수 실행됨")  # 로그

    images = request.FILES.getlist('image')
    if not images or len(images) != 5:
        return Response({'detail': '이미지 5장이 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
    os.makedirs(upload_dir, exist_ok=True)

    print(f"[MEDIA_ROOT 경로 확인]: {settings.MEDIA_ROOT}")
    print(f"[업로드 대상 경로 확인]: {upload_dir}")

    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    valid_face_count = 0
    results = []

    for idx, img_file in enumerate(images):
        img_array = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        filename = f"{prefix}_{idx}.jpg"
        filepath = os.path.join(upload_dir, filename)

        # 디코딩 확인
        if image is None:
            print(f"❌ 이미지 디코딩 실패: {img_file.name}")
            continue
        else:
            print(f"✅ 디코딩 성공: {img_file.name}, shape={image.shape}")

        # 이미지 저장
        success = cv2.imwrite(filepath, image)
        print(f"[이미지 저장 성공 여부]: {success}")
        print(f"[실제 저장 경로]: {filepath}")

        if os.path.exists(filepath):
            print(f"📁 파일 존재 확인됨: {filepath}")
        else:
            print(f"⚠️ 파일 없음 (저장 실패했을 수도 있음): {filepath}")

        # 얼굴 검사
        face_ok = is_face_present(image)
        blur_ok = not is_blurry(image)
        frontal_ok = is_frontal_face(image) if face_ok else False

        result = {
            "filename": filename,
            "face_detected": face_ok,
            "not_blurry": blur_ok,
            "frontal": frontal_ok
        }
        results.append(result)

        if face_ok and blur_ok and frontal_ok:
            valid_face_count += 1

    if valid_face_count == 0:
        return Response({
            "detail": "5장 모두에서 적합한 얼굴 사진을 찾지 못했습니다.",
            "results": results
        }, status=422)

    return Response({
        "detail": "적어도 한 장 이상의 유효한 얼굴 사진이 있습니다.",
        "results": results
    }, status=200)
