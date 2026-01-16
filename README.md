# Detecția Cancerului de Sân cu ResNet50 și CLAHE (CBIS-DDSM)

Acest proiect implementează o soluție bazată pe **Deep Learning** pentru clasificarea mamografiilor în categorii **Benigne** sau **Maligne**. Soluția utilizează arhitectura **ResNet50** și o etapă de preprocesare cu **CLAHE (Contrast Limited Adaptive Histogram Equalization)** pentru a evidenția detaliile tumorale.

---

## 1. Structura Proiectului

Pentru ca scripturile să funcționeze corect, asigurați-vă că structura folderelor arată exact așa:

```text
project/
│
├── best_breast_cancer_model.keras   <-- Modelul antrenat (trebuie să fie aici, direct în project)
├── breast_cancer_resnet_model.keras  <-- Modelul antrenat pentru testare parametri
│
├── csv/                             <-- Folder cu datele tabelare
│   ├── mass_case_description_train_set.csv
│   ├── mass_case_description_test_set.csv
│   └── dicom_info.csv
│
├── jpeg/                            <-- Dataset-ul dezarhivat (foldere cu UID-uri lungi)
│   └── 1.3.6.1.4.1.9590...
│
├── testing_img/                     <-- FOLDER NOU (creat manual pentru demo)
│   ├── 1-267.jpg
│   ├── 1-071.jpg
│   ├── 1-111.jpg
│   ├── test_benign_new.jpg
│   └── test_malign_new.jpg
│
├── prepare.py                       <-- Modul preprocesare CLAHE
├── train.py                         <-- Model mai vechi pe care am încercat parametrii
├── train2.py                        <-- Script antrenare model
├── evaluate_final.py                <-- Script generare grafice (Matrice Confuzie, ROC)
└── ground_truth.py                  <-- Script testare manuală (Demo Vizual)
```

## 📥 Descărcare Dataset

Deoarece setul de date CBIS-DDSM este prea mare pentru GitHub, acesta trebuie descărcat separat.

1.  **Descărcați arhiva cu datele (Imagini + CSV)** de aici si modelele:
    https://drive.google.com/file/d/1JDBy4OOkg-_jsQ7ATKcjVBzug0JpG4rC/view?usp=sharing

2.  Dezarhivați conținutul în folderul `project`, astfel încât să aveți structura:
    - `project/jpeg/...`
    - `project/csv/...`

---

## 2. Instalare Dependențe

Aveți nevoie de **Python 3.1.1** și următoarele biblioteci instalate. Rulați în terminal:

```bash
pip install tensorflow pandas numpy opencv-python matplotlib seaborn scikit-learn
```

---

## 3. Configurare Critică (Căile din Cod)

Deoarece scripturile conțin căi **absolute** către fișierele de pe disc, trebuie să modificați variabila `BASE_DIR` sau `PROJECT_DIR` în următoarele fișiere înainte de rulare:

- `ground_truth.py`
- `evaluate_final.py`
- `train2.py`

### Pași:

1. Deschideți fișierele menționate mai sus.
2. Căutați linia de la început:

```python
BASE_DIR = r'C:\Users\Christiana\Desktop\KBS'
# sau
PROJECT_DIR = ...
```

3. Modificați calea astfel încât să pointeze exact către folderul unde ați descărcat proiectul pe calculatorul dumneavoastră.

---

## 4. Pregătirea Imaginilor pentru Testare (Ground Truth)

Pentru a rula scriptul de demonstrație `ground_truth.py`, folosesc **5 imagini specifice** din dataset-ul mare (`jpeg`) în folderul `testing_img`.

---

## 5. Rularea Scripturilor

### A. Demonstrație Vizuală (`ground_truth.py`)

#### Comandă:

```bash
python ground_truth.py
```

#### Rezultat:

- Se vor deschide ferestre cu imaginile analizate
- **Titlu VERDE** → Modelul a prezis corect
- **Titlu ROȘU** → Modelul a prezis greșit

---

### B. Evaluare Statistică (`evaluate_final.py`)

Generează raportul complet de performanță pe setul de testare (378 imagini).

#### Comandă:

```bash
python evaluate_final.py
```

#### Rezultat:

- Afișează **Matricea de Confuzie (Heatmap)**
- Afișează **Curba ROC și scorul AUC**
- Printează raportul text:
  - Precision
  - Recall
  - F1-Score

---

### C. Antrenare Model (`train2.py`)

Dacă doriți să re-antrenați modelul de la zero. (no doriți, durează ceva timp)

#### Comandă:

```bash
python train2.py
```

#### Detalii:

- Folosește **Class Weights** pentru balansarea datelor
- Aplică **augmentare** (rotire, zoom) pentru a preveni overfitting-ul
- Modelul final este salvat automat ca:

```
best_breast_cancer_model.keras
```

în folderul principal al proiectului.

---

## 7. Alte link-uri pentru alte imagini de testare

```
.../jpeg/1.3.6.1.4.1.9590.100.1.2.195619769212745505323965034531436697402
.../jpeg/1.3.6.1.4.1.9590.100.1.2.128610930012277969524545675822048951667
.../jpeg/1.3.6.1.4.1.9590.100.1.2.233363484412206214942278982962693471990
.../jpeg/1.3.6.1.4.1.9590.100.1.2.228321467711661695217126616462194081040
```

---

## 8. Rezultate Vizuale (Imagini Salvate)

Dacă ați rula `ground_truth.py` și `evaluate_final.py`, ați vedea imaginile din:

```
project/
└── photos_with_results/
    ├── confusion_matrix.png
    ├── my_clahe.png
    ├── precision_recall_f1_support.png
    ├── rezultat_model_1.png
    ├── rezultat_model_2.png
    ├── rezultat_model_3.png
    ├── rezultat_model_4.png
    ├── rezultat_model_5.png
    └── roc_curve.png
```

---

## 9. Note

- Asigurați-vă că aveți suficient spațiu pe disc pentru dataset
- Recomandat: rulare pe sistem cu GPU pentru antrenare mai rapidă
- Testarea și evaluarea pot rula și pe CPU
