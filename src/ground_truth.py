# TESTARE MODEL:
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. CONFIGURARE ---
# !!!!!!!! SCHIMBATI AICI CALEA PENTRU TESTARE (doar directorul) !!!!!!!!
BASE_DIR = r'C:\Users\Christiana\Desktop\KBS'
PROJECT_DIR = os.path.join(BASE_DIR, 'project')
TEST_IMG_FOLDER = os.path.join(PROJECT_DIR, 'testing_img')
MODEL_PATH = os.path.join(PROJECT_DIR, 'best_breast_cancer_model.keras')
IMG_SIZE = (224, 224)

# --- 2. MAPAREA MANUALĂ A ADEVĂRULUI (GROUND TRUTH) ---
GROUND_TRUTH_MAP = {
    # Nume fișier       : (Eticheta Reală,      Detalii Extra)
    "1-267.jpg":          ("MALIGNANT",         "Mass-Test_P_00741"),
    "1-071.jpg":          ("BENIGN",            "Mass-Test_P_01257"),
    "1-111.jpg":          ("BENIGN",            "Mass-Test_P_00114 (Benign_Without_Callback)"),
    "1-032.jpg":          ("BENIGN",            "Mass-Test_P_01090 (Right MLO)"),
    "1-252.jpg":          ("MALIGNANT",         "Mass-Test_P_00656 (Right MLO)"),
    
}

# --- 3. FUNCȚII ---
def preprocessing_function_rahman(img):
    img = img.astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return preprocess_input(img_final.astype('float32'))

# --- 4. PREDICTOR ---
print(f"Încărcare model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("EROARE: Nu găsesc modelul!")
    exit()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Eroare load model: {e}")
    exit()

files = [f for f in os.listdir(TEST_IMG_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
print(f"\nAm găsit {len(files)} imagini. Încep testarea...\n")

for filename in files:
    img_path = os.path.join(TEST_IMG_FOLDER, filename)
    
    if filename in GROUND_TRUTH_MAP:
        real_label, details = GROUND_TRUTH_MAP[filename]
    else:
        real_label = "NECUNOSCUT"
        details = "Fișierul nu e în lista de mapare manuală"

    # Predicție
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_pre = preprocessing_function_rahman(cv2.resize(img_rgb, IMG_SIZE))
    score = model.predict(np.expand_dims(img_pre, axis=0), verbose=0)[0][0]
    
    pred_label = "MALIGNANT" if score > 0.5 else "BENIGN"
    
    # Verificare Corectitudine
    status_color = 'black'
    status_icon = "?"
    
    if real_label != "NECUNOSCUT":
        if pred_label == real_label:
            status_color = 'green'
            status_icon = "CORECT"
        else:
            status_color = 'red'
            status_icon = " GREȘIT"

    print(f"Imagine: {filename}")
    print(f"  > Sursă:     {details}")
    print(f"  > REALITATE: {real_label}")
    print(f"  > PREDICȚIE: {pred_label} (Scor: {score:.4f})")
    print(f"  > REZULTAT:  {status_icon}")
    print("-" * 40)

    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    title_text = f"{filename}\nReal: {real_label}\nPred: {pred_label} ({score:.2f})"
    plt.title(title_text, color=status_color, fontweight='bold')
    plt.axis('off')
    plt.show()