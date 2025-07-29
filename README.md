# 🔬 Arabic Medical Text Classifier

A Natural Language Processing (NLP) project designed to classify Arabic medical complaints into their appropriate medical specialties.

The goal is to help users (or digital systems) identify which medical department they should consult based on a short free-text description of symptoms written in Arabic, either colloquial or formal.

---

## 🚀 Project Overview

**Example Input:**
> "بحس بتنميل في رجلي ومش قادر أمشي كويس"  
**Output:**
> **Neurology**

---

## 🧠 Components

### 📊 Dataset
- Manually curated dataset of Arabic symptom descriptions.
- Each sentence is labeled with one of 15 medical specialties:
  - Cardiology, Neurology, Gynecology, ENT, Dentistry, Ophthalmology, Internal Medicine, etc.
- Each class has **200 examples** to ensure data balance.

### 🧹 Data Preprocessing
- Lowercasing
- Removing digits and punctuation
- Stopword removal (lightly filtered to preserve meaning)
- Custom Arabic cleaning logic

### 🧠 Models
- **Phase 1**: `LogisticRegression` with `TfidfVectorizer`


### 📈 Evaluation
- Used `classification_report` for accuracy, precision, recall, and F1-score
- Stored:
  - `label_encoder.pkl`
  - `tokenizer/`
  - `marbert_model/`

---

## 🛠️ How to Use

### 1. Train the Model:
```bash
python train_marbert_classifier.py
```

🖼️ User Interface

    The project includes a frontend written in HTML, CSS, and Vanilla JS
    
    Users can type their symptoms
    
    On submit, a POST request is sent to /predict endpoint
    
    The predicted specialty is shown instantly on the page
