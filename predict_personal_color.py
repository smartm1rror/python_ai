import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# 퍼스널 컬러 예측 모델 
# 입력 이미지에서 얼굴을 감지하여 중심을 crop
# 피부색 마스킹(YCrCb 기반) 후 ResNet 모델로 퍼스널컬러 예측

# 경로 설정
MODEL_PATH = os.path.join('saved_models', 'personal_color_resnet.h5')
CLASS_PATH = os.path.join('saved_models', 'class_names.txt')
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 모델 및 클래스 이름 로딩 함수
def load_model_and_classes():
    #학습된 모델과 클래스 이름 목록을 불러옴.
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        sys.exit()

    if not os.path.exists(CLASS_PATH):
        print(f"클래스 파일이 없습니다: {CLASS_PATH}")
        sys.exit()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip().replace("\\", "_") for line in f]

    print(f"클래스 목록 로드됨: {class_names}")
    return model, class_names

# 얼굴 중심 crop 함수(openCV 기반)
def crop_face(img):
    #입력 이미지에서 얼굴을 찾아 중심 영역만 crop
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("얼굴 미검출 - 원본 사용")
        return img

# 가장 큰 얼굴을 기준으로 crop
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    margin = int(0.2 * h)
    cropped = img[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
    return cropped

# 피부 영역만 추출하는 마스크 적용 함수 
def apply_skin_mask(img):

    # RGB 이미지를 YCrCb 색공간으로 변환 후 피부색 범위에 해당하는 부분만 마스크로 추출
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # 마스크 정제(노이즈 제거)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # 마스크 영역만 추출
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# 퍼스널 컬러 예측 함수
def predict_personal_color(image_path, model, class_names):
    if not os.path.exists(image_path):
        return "이미지가 없습니다.", None

    img = cv2.imread(image_path)
    if img is None:
        return "이미지를 불러올 수 없습니다.", None

# 얼굴 감지 → crop → 피부 마스크 → 전처리
    img = crop_face(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = apply_skin_mask(img)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

# 예측 수행
    pred = model.predict(img)
    print(f"예측 결과 배열: {pred}")
    print(f"예측 결과 shape: {pred.shape}")

# 예측 결과 검증
    if pred.ndim != 2 or pred.shape[0] != 1 or pred.shape[1] != len(class_names):
        return "예측 오류", None

# 결과 반환
    predicted_index = np.argmax(pred)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(pred)) * 100

    return predicted_class, confidence

# 실행 시작
if __name__ == '__main__':
    test_image = 'test.jpg'
    print(f"예측 시작: {test_image}")

    model, class_names = load_model_and_classes()
    result, confidence = predict_personal_color(test_image, model, class_names)

    if confidence is not None:
        print(f"예측 결과: {result} (신뢰도: {confidence:.2f}%)")
    else:
        print(f"예측 실패: {result}")
