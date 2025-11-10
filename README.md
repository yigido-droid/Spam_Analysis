# 📈 Does Removing Stopwords Change TF-IDF Spam Classification Results?

This project analyzes how **removing stopwords** (common, non-informative words) affects  
the performance of a **spam message classifier** built using **TF-IDF vectorization**  
and the **Multinomial Naive Bayes** algorithm.

---

## 📁 Dataset

- **Source file:** `spam.csv`  
- **Generated files:**
  - `spam_clean.csv` → Removed empty rows and columns  
  - `spam_clean_ready.csv` → Cleaned text (URLs, punctuation removed)  
  - `spam_clean_removed_stopwords.csv` → Cleaned text + Stopwords removed  

| Label | Count |
|:------|------:|
| ham   | 4825  |
| spam  | 747   |

---

## 🧰 Libraries Used

```python
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```
## 🧪 Version 1 — Without Stopword Removal

The following results correspond to the **TF-IDF + Multinomial Naive Bayes** model  
trained on the `v3_cleaned` column — where stopwords were **not removed**.

### Confusion matrix:
<img width="535" height="448" alt="Ekran Resmi 2025-11-08 12 50 00" src="https://github.com/user-attachments/assets/0c42aec6-693f-4647-b82e-de9abe9b820f" />

---

### ✅ Model Performance
<img width="465" height="243" alt="Ekran Resmi 2025-11-08 12 49 34" src="https://github.com/user-attachments/assets/7a75a64d-45ba-403e-a812-153c1096daae" />


### 🧩 Interpretation:
Model without stopword removal achieved **96.6% accuracy**.  
Slight confusion between *ham* and *spam* (30 FP, 8 FN),  
but overall classification remained highly reliable.

---

## 🧪 Version 2 — With Stopword Removal

This version uses the **TF-IDF + Multinomial Naive Bayes** model trained on  
the `key_column` text field — where **stopwords were removed** before vectorization.

---

### Confusion matrix:

<img width="536" height="447" alt="Ekran Resmi 2025-11-08 12 49 00" src="https://github.com/user-attachments/assets/5985b2fe-163b-4b1f-a775-fc8c86406d56" />

---

### ✅ Model Performance

<img width="469" height="246" alt="Ekran Resmi 2025-11-08 12 48 03" src="https://github.com/user-attachments/assets/5189638f-514d-41dc-a6f1-821947b22a63" />

### 🧩 Interpretation:  
After removing stopwords, accuracy slightly improved to **96.9%**.  
Spam F1-score rose to **0.89**, showing cleaner text helped the model  
capture spam patterns a bit more effectively.

## 📊 Overall Report

Both experiments — with and without stopword removal — produced **highly consistent results**.  
Removing stopwords led to a **minor accuracy improvement** from **96.59% → 96.86%**,  
and a small boost in spam **F1-score (0.88 → 0.89)**.

| Version | Stopwords | Accuracy | Spam F1-Score | Observation |
|:--------|:-----------|:----------:|:---------------:|:-------------|
| **V1** | Kept | 0.9659 | 0.88 | Model performs strongly even with stopwords present |
| **V2** | Removed | **0.9686** | **0.89** | Slight improvement due to reduced noise |

**Conclusion:**  
Removing stopwords provided a **small but measurable gain** in model precision and F1-score,  
especially for spam detection. However, the minimal difference suggests that  
**TF-IDF weighting already mitigates most stopword influence**,  
making the model robust in both cases.
