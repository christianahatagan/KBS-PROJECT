# PREPROCESARE CU CLAHE: SCHIMBATI LA SEMNUL EXCLAMARII DE JOS PENTRU A TESTA (link-uri de testare sunt in README)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image_rahman(image_path, target_size=(224, 224)):
    """
    Implementeaza pipeline-ul de preprocesare inspirat de Rahman et al.[cite: 5]:
    1. Citire imagine
    2. Redimensionare
    3. Aplicare CLAHE (Contrast Enhancement)
    4. Normalizare pentru ResNet-50
    """
    # 1. Citire imagine 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None

    # 2. Redimensionare la standardul ResNet (224x224)
    img_resized = cv2.resize(img, target_size)
    
    # 3. APLICARE CLAHE 
    # ClipLimit=2.0 și TileGridSize=(8,8) sunt valori standard bune
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
    
    # 4. Convertire la 3 canale (ResNet așteapta RGB, chiar daca e mamografie)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # 5. Preprocesare specifică ResNet (normalizare)
    img_final = preprocess_input(img_rgb)
    
    return img_rgb, img_final # Returnăm și img_rgb doar pentru vizualizare

# !!!!!!!!!! 
original, preprocessed = preprocess_image_rahman(r'C:\Users\Christiana\Desktop\KBS\project\jpeg\1.3.6.1.4.1.9590.100.1.2.499558611862523307025745211397332529\1-036.jpg')
plt.imshow(original)
plt.show()