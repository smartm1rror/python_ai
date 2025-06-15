import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# 데이터 및 모델 저장 경로 설정
DATA_DIR = 'data/personal_color'  # 하위 폴더: autumn, spring, summer, winter
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# 학습용 데이터 전처리 및 증강 설정 (train 데이터에만 증강 적용)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=5,
    brightness_range=[0.9, 1.1],
    width_shift_range=0.02,
    height_shift_range=0.02
)

# 검증용 데이터 전처리 (증강 없이 단순 스케일링)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 학습 데이터 로딩
train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# 검증 데이터 로딩
val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 클래스 이름 저장 (autumn, spring, summer, winter)
class_names = [k.replace('\\', '_') for k in train_data.class_indices.keys()]
with open(os.path.join(SAVE_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"클래스 목록 저장 완료: {class_names}")

# 클래스 불균형 보정용 가중치 계산
labels = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("클래스 가중치:", class_weights)

# 사전 학습된 ResNet50 모델을 기반으로 특성 추출 및 분류기 구성
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

 #전체 모델 중 마지막 100개 레이어만 학습 가능하도록 설정 (Fine-Tuning)
for layer in base_model.layers[:-100]:
    layer.trainable = False
for layer in base_model.layers[-100:]:
    layer.trainable = True

# 모델 컴파일 (손실 함수, 최적화 알고리즘, 평가지표 설정)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 저장 및 조기 종료 설정
checkpoint = ModelCheckpoint(
    os.path.join(SAVE_DIR, 'best_model_resnet.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 모델 학습 수행
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# 최종 모델 저장
model.save(os.path.join(SAVE_DIR, 'personal_color_resnet.h5'))
print("ResNet50 기반 모델 학습 및 저장 완료")
