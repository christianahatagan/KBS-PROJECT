import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import cv2

# --- 1. CONFIGURARE ---
# !!!!!!!!!! PENTRU TESTARE VA ROG SA SCHIMBATI CALEA AICI
MODEL_PATH = r'C:\Users\Christiana\Desktop\KBS\project\best_breast_cancer_model.keras'

# !!!!!!!!!! PENTRU TESTARE VA ROG SA SCHIMBATI CALEA AICI:
PROJECT_DIR = r'C:\Users\Christiana\Desktop\KBS\project'
CSV_DIR = os.path.join(PROJECT_DIR, 'csv')
IMG_DIR = os.path.join(PROJECT_DIR, 'jpeg')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# --- 2. PREGĂTIREA DATELOR (Doar Test Set) ---
import pandas as pd
test_csv = os.path.join(CSV_DIR, 'mass_case_description_test_set.csv')
dicom_info_csv = os.path.join(CSV_DIR, 'dicom_info.csv')

def prepare_test_data(csv_path, dicom_csv_path):
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

print("Se încarcă datele de test...")
df_test = prepare_test_data(test_csv, dicom_info_csv)

# --- 3. PREPROCESARE (Aceeași ca la antrenare) ---
def preprocessing_function_rahman(img):
    img = img.astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return preprocess_input(img_final.astype('float32'))

test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function_rahman)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col='final_path',
    y_col='pathology',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  
)

# --- 4. EVALUARE ---
if not os.path.exists(MODEL_PATH):
    print(f"EROARE: Nu găsesc modelul la {MODEL_PATH}")
    print("Verifică dacă fișierul există acolo!")
else:
    print(f"\nSe încarcă modelul: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print("Se generează predicții pe tot setul de test...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype("int32").flatten()
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys()) # ['BENIGN', 'MALIGNANT']

    # --- A. MATRICEA DE CONFUZIE ---
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show() 

    # --- B. RAPORT DETALIAT ---
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # --- C. CURBA ROC ---
    fpr, tpr, thresholds = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show() 

    print(f"AUC FINAL: {roc_auc}")