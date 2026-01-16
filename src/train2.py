import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- 1. CONFIGURARE ---
BASE_DIR = r'C:\Users\Christiana\Desktop\KBS'
if not os.path.exists(BASE_DIR):
    print(f"ATENȚIE: Folderul {BASE_DIR} nu există! Îl creez acum.")
    os.makedirs(BASE_DIR)

PROJECT_DIR = os.path.join(BASE_DIR, 'project')
SAVE_PATH = os.path.join(BASE_DIR, 'best_breast_cancer_model.keras')

CSV_DIR = os.path.join(PROJECT_DIR, 'csv')
IMG_DIR = os.path.join(PROJECT_DIR, 'jpeg')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16 
EPOCHS = 30     

# --- 2. PREGĂTIRE DATE ---
def prepare_data(csv_path, dicom_csv_path):
    df = pd.read_csv(csv_path)
    dicom_info = pd.read_csv(dicom_csv_path)
    
    def extract_uid(path):
        try: return path.split('/')[-2]
        except: return None
    df['SeriesUID'] = df['image file path'].apply(extract_uid)
    
    merged_df = pd.merge(df, dicom_info, left_on='SeriesUID', right_on='SeriesInstanceUID', how='inner')
    
    def clean_path(path):
        relative_path = path.replace('CBIS-DDSM/jpeg', '').strip('/\\')
        return os.path.join(IMG_DIR, relative_path)

    merged_df['final_path'] = merged_df['image_path'].apply(clean_path)
    merged_df['pathology'] = merged_df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
    return merged_df

train_csv = os.path.join(CSV_DIR, 'mass_case_description_train_set.csv')
test_csv = os.path.join(CSV_DIR, 'mass_case_description_test_set.csv')
dicom_info_csv = os.path.join(CSV_DIR, 'dicom_info.csv')

print("Se pregătesc datele...")
df_train = prepare_data(train_csv, dicom_info_csv)
df_test = prepare_data(test_csv, dicom_info_csv)

# --- 3. PREPROCESARE ---
def preprocessing_function_rahman(img):
    img = img.astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return preprocess_input(img_final.astype('float32'))

# --- 4. GENERATORI ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function_rahman,
    horizontal_flip=True, vertical_flip=True, rotation_range=30,    
    zoom_range=0.2, shear_range=0.1, width_shift_range=0.1,
    height_shift_range=0.1, fill_mode='nearest'
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function_rahman)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train, x_col='final_path', y_col='pathology',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
)
test_generator = val_datagen.flow_from_dataframe(
    dataframe=df_test, x_col='final_path', y_col='pathology',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# --- 5. MODEL ---
def build_advanced_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = True
    for layer in base_model.layers[:140]:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'), 
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'), 
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

model = build_advanced_model()

# --- 6. CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max', verbose=1),
    ModelCheckpoint(SAVE_PATH, monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

# --- 7. ANTRENARE ---
print(f"Modelul se va salva în: {SAVE_PATH}")
train_classes = train_generator.classes
class_weights_list = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(train_classes), y=train_classes
)
class_weights = dict(enumerate(class_weights_list))

try:
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights 
    )
except Exception as e:
    print(f"Eroare la antrenare: {e}")

