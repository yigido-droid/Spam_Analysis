# 📩 Spam Message Detection using Naive Bayes & CountVectorizer


## 🧠 Project Overview
This project builds a **Spam Message Classifier** that automatically detects whether an SMS message is **spam** or **ham (not spam)**.  
The pipeline includes **data cleaning**, **stopword removal**, **feature extraction** using **CountVectorizer**, and **model training** with **Multinomial Naive Bayes**.

The goal is to demonstrate a full **Natural Language Processing (NLP)** pipeline — from raw text to model evaluation — using the **SMS Spam Collection Dataset**.

---

## 📂 Dataset
The dataset used is the **SMS Spam Collection Dataset**, containing **5,572 messages** labeled as *ham* or *spam*.

| Label | Count |
|:------|------:|
| Ham   | 4,825 |
| Spam  |   747 |

After cleaning and processing, the data was saved in several stages:
- `spam_clean.csv` – base cleaned version  
- `spam_clean_ready.csv` – cleaned and normalized text  
- `spam_clean_removed_stopwords.csv` – version without stopwords  

---

## 🧹 Text Preprocessing
The cleaning pipeline applies:

1. Lowercasing  
2. Removing URLs, emails, phone numbers  
3. Removing punctuation and special characters  
4. Trimming extra whitespace  
5. (Optional) Removing English stopwords (`nltk.corpus.stopwords`)  

Example transformation:

| Original Text | Cleaned Text |
|:---------------|:-------------|
| “Go until jurong point, crazy.. Available only in bugis n great world…” | “go jurong point crazy available bugis n great world” |

---

## 🧮 Feature Extraction
Text features were generated using **CountVectorizer** with the following settings:

```python
CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=50000
)
```

This builds a **Bag-of-Words** model capturing both unigrams and bigrams.

---

## 🧠 Model Training
The model used is **Multinomial Naive Bayes**.  
Training and test data were split 80/20 with stratification.  
Class imbalance was handled via sample weighting:

```python
clf.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight="balanced", y=y_train))
```

---

## 📊 Evaluation Results

### 1️⃣ Without Stopword Removal
| Metric | Ham | Spam | Macro Avg | Weighted Avg |
|:--|--:|--:|--:|--:|
| Precision | 0.99 | 0.86 | 0.92 | 0.97 |
| Recall | 0.98 | 0.94 | 0.96 | 0.97 |
| F1-Score | 0.98 | 0.90 | 0.94 | 0.97 |
| **Accuracy** |  |  |  | **0.9713** |

Confusion Matrix:
```
[[943  23]
 [  9 140]]
```

---

### 2️⃣ With Stopword Removal
| Metric | Ham | Spam | Macro Avg | Weighted Avg |
|:--|--:|--:|--:|--:|
| Precision | 0.99 | 0.86 | 0.92 | 0.97 |
| Recall | 0.98 | 0.93 | 0.95 | 0.97 |
| F1-Score | 0.98 | 0.89 | 0.94 | 0.97 |
| **Accuracy** |  |  |  | **0.9704** |

Confusion Matrix:
```
[[943  23]
 [ 10 139]]
```

---

## 🔍 Comparison: With vs Without Stopwords
| Aspect | Without Stopwords | With Stopwords |
|:--|:--|:--|
| Accuracy | 97.13% | 97.04% |
| Spam Recall | 0.94 | 0.93 |
| Spam F1-Score | 0.90 | 0.89 |
| Vocabulary Size | Larger (includes common words) | Smaller (stopwords removed) |
| Model Stability | Slightly better recall on spam | Slightly faster training |

👉 **Observation:**  
Removing stopwords slightly reduced accuracy (by 0.001) and spam recall, but simplified the feature space and reduced training complexity.  
Since spam messages often contain *common words used in marketing phrases*, removing stopwords can sometimes eliminate weak but helpful context words — explaining the small performance drop.

---

## 🧾 File Structure

```
📁 Datasets/
 ├── spam.csv
 ├── spam_clean.csv
 ├── spam_clean_ready.csv
 └── spam_clean_removed_stopwords.csv
```
---
## ✨ Author
**Yiğit Can Kınalı**  
