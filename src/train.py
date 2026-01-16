import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras import layers, models, optimizers

# --- 1. CONFIGURARE CĂI (Updatează dacă e nevoie) ---
PROJECT_DIR = r'C:\Users\Christiana\Desktop\KBS\project'
CSV_DIR = os.path.join(PROJECT_DIR, 'csv')   
IMG_DIR = os.path.join(PROJECT_DIR, 'jpeg')  

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. FUNCȚIE PENTRU PREGĂTIREA DATAFRAME-ULUI ---
def prepare_data(csv_path, dicom_csv_path):
    """
    Citește CSV-ul de antrenament și îl combină cu dicom_info pentru a găsi calea imaginilor.
    """
    print(f"Încărcăm datele din: {csv_path}")
    
    # 1. Citim CSV-urile
    df = pd.read_csv(csv_path)
    dicom_info = pd.read_csv(dicom_csv_path)
    
    # 2. Extragem UID-ul Seriei din calea lungă a fișierului DICOM
    # SeriesUID este penultimul element
    def extract_uid(path):
        try:
            return path.split('/')[-2]
        except:
            return None
            
    df['SeriesUID'] = df['image file path'].apply(extract_uid)
    
    # 3. unim datele clinice (df) cu informațiile despre fișiere (dicom_info)
    # Folosim UID-ul ca cheie comună
    merged_df = pd.merge(df, dicom_info, left_on='SeriesUID', right_on='SeriesInstanceUID', how='inner')
    
    # 4. Construim calea locală reală către imaginea JPG
    # dicom_info['image_path'] arată ca: CBIS-DDSM/jpeg/1.3.6.../1-036.jpg
    def clean_path(path):
        # Scoatem 'CBIS-DDSM/jpeg' din cale și o lipim la folderul nostru local
        relative_path = path.replace('CBIS-DDSM/jpeg', '').strip('/\\')
        full_path = os.path.join(IMG_DIR, relative_path)
        return full_path

    merged_df['final_path'] = merged_df['image_path'].apply(clean_path)
    
    # 5. Curățăm etichetele (Labels)
    # Convertim 'BENIGN_WITHOUT_CALLBACK' în 'BENIGN' pentru clasificare binară
    merged_df['pathology'] = merged_df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
    
    print(f"  -> Am găsit {len(merged_df)} imagini valide.")
    return merged_df

# Încărcăm datele
train_csv = os.path.join(CSV_DIR, 'mass_case_description_train_set.csv')
test_csv = os.path.join(CSV_DIR, 'mass_case_description_test_set.csv')
dicom_info_csv = os.path.join(CSV_DIR, 'dicom_info.csv')

df_train = prepare_data(train_csv, dicom_info_csv)
df_test = prepare_data(test_csv, dicom_info_csv)

# Verificăm dacă fișierele există fizic (pentru debug)
if not os.path.exists(df_train['final_path'].iloc[0]):
    print(f"\n[EROARE] Nu găsesc fișierul: {df_train['final_path'].iloc[0]}")
    print("Verifică variabila IMG_DIR de la începutul scriptului!")
    exit()

# --- 3. IMPLEMENTARE CLAHE (Metoda Rahman) ---
def preprocessing_function_rahman(img):
    img = img.astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return preprocess_input(img_final.astype('float32'))

# --- 4. GENERATOARE DE DATE (FLOW FROM DATAFRAME) ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function_rahman,
    horizontal_flip=True,
    rotation_range=20
)

# Generator pentru Antrenare
train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='final_path',      # Coloana cu calea imaginii
    y_col='pathology',       # Coloana cu eticheta (BENIGN/MALIGNANT)
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Generator pentru Testare/Validare
test_generator = datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col='final_path',
    y_col='pathology',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 5. MODELUL (RESNET-50) ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# --- 6. ANTRENARE ---
print("\n--- Începe antrenamentul ---")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20  
)

model.save('breast_cancer_resnet_model.h5')
