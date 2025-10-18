# 🔧 Air Compressor Fault Detection

This project can automatically **detect faults in an air compressor** by listening to the **sound** it makes.  
It uses **machine learning** and **audio feature extraction** to tell whether your compressor is healthy or has a fault like:
- Bearing issue  
- Flywheel problem  
- Inlet/Outlet valve leak  
- Non-return valve fault  
- Piston or rider belt issue  

---

## 🚀 How It Works

1. The compressor sound is recorded as a `.wav` file.  
2. The program extracts sound features like:
   - RMS (Root Mean Square)
   - ZCR (Zero Crossing Rate)
   - MFCC (Mel-frequency coefficients)
3. These features are scaled and fed into a **trained ML model (SVM)**.  
4. The model predicts which type of fault the sound belongs to.  
5. You can upload multiple files using a **Streamlit web app**, and download the results as a CSV.

---

## 🧠 Technologies Used

| Tool | Purpose |
|------|----------|
| **Python** | Main programming language |
| **librosa** | Extracts audio features |
| **numpy & pandas** | Handles numbers and data |
| **scikit-learn** | Trains and runs the ML model |
| **joblib** | Saves/loads trained models |
| **streamlit** | Web-based interface |
| **tqdm** | Progress bar in terminal |
| **python-docx** | Creates Word reports |

---

## 📁 Folder Structure

```
Fault detection/
├── app.py                  # Streamlit web app
├── main.py                 # Command-line version
├── train.py                # Model training script
├── feature_extraction.py   # Extracts sound features
├── models/                 # Saved model and scaler
├── data/                   # Audio dataset (split into 4 parts)
├── reports/                # Project reports (Word files)
└── README.md               # This file
```

---

## ⚙️ How to Use

### Step 1: Clone or Download the Project
```
git clone https://github.com/0wlReaper/Air-Compressor-Fault-Detection.git
```

### Step 2: Open the Folder
```
cd Air-Compressor-Fault-Detection
```

### Step 3: Install Required Libraries
```
pip install -r requirements.txt
```

### Step 4: Run the Streamlit App
```
streamlit run app.py
```

### Step 5: Upload Compressor Audio Files
- Upload one or more `.wav` recordings.
- The app will predict the type of fault.
- You can also download the results as a CSV.

---

## 📊 Model Info

- **Model:** SVM (Support Vector Machine)
- **Accuracy:** Around 99% on test data
- **Data Type:** Audio recordings (WAV)
- **Feature Set:** MFCC, RMS, ZCR, spectral features

---

## 🧑‍💻 Author

**Sivadev**  
GitHub: [@0wlReaper](https://github.com/0wlReaper)

---

> 🗣️ “Machines can listen too — if you teach them how.”

---
