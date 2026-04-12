# 🧠 Emotion-Aware Reflective Journal System

An NLP + Machine Learning system that analyzes **journal entries** and contextual lifestyle signals to detect a user's **emotional state**, predict **intensity**, and suggest **personalized mental wellness actions** — built using TF-IDF, Random Forest Classifier, and Random Forest Regressor.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [NLP Pipeline](#nlp-pipeline)
- [Model Details](#model-details)
- [Decision Engine](#decision-engine)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 📖 Overview

This project goes beyond basic sentiment analysis — it simulates a **reflective journaling assistant** that:

1. Takes a user's **journal text + lifestyle context** as input
2. Detects the **emotional state** (happy, sad, angry, neutral, etc.)
3. Predicts the **intensity** of that emotion (numeric score)
4. Runs a **Decision Engine** that recommends a **personalized action** with timing

**Real-World Use Cases:**
- Mental health journaling apps
- Mood tracking systems
- Employee wellness monitoring
- AI-powered therapy assistants

---

## ⚙️ How It Works

```
Journal Text + Context Features
        │
        ▼
  Feature Combination
  (text + ambience + time_of_day +
   previous_day_mood + face_emotion_hint +
   reflection_quality)
        │
        ▼
  TF-IDF Vectorization (max 5000 features)
        │
        ├──► RandomForestClassifier ──► Predicted Emotion Label
        │                               (happy / sad / angry / neutral...)
        │
        └──► RandomForestRegressor  ──► Predicted Intensity Score
                                        (numeric value)
                                              │
                                              ▼
                                     🔧 Decision Engine
                                              │
                                              ▼
                               Suggested Action + When To Do It
```

---

## ✨ Features

- 📓 Multi-feature input — journal text + ambience, time of day, mood, face hint
- 🤖 Dual ML model — Classifier (emotion label) + Regressor (intensity score)
- 🔧 Rule-based Decision Engine — maps emotion + intensity → actionable suggestions
- 💬 Real-time User Input System — type your feelings, get instant recommendations
- 📁 Batch prediction — processes full test dataset and saves `final_output.csv`
- 🧹 Text preprocessing with TF-IDF vectorization

---

## 🛠 Tech Stack

| Category          | Tools / Libraries                              |
|-------------------|------------------------------------------------|
| Language          | Python 3.x                                     |
| ML Models         | RandomForestClassifier, RandomForestRegressor  |
| NLP               | TF-IDF Vectorizer (Scikit-learn)               |
| Preprocessing     | LabelEncoder, fillna, feature combination      |
| Data Processing   | Pandas, NumPy                                  |
| Model Evaluation  | accuracy_score, mean_squared_error             |
| Development Env   | Jupyter Notebook / VS Code                     |
| Version Control   | Git, GitHub                                    |

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `Sample_arvyax_reflective_dataset.xlsx` | Training data with labeled emotions |
| `arvyax_test_inputs_120.xlsx` | Test data for batch predictions |

### Input Features Used:

| Feature               | Description                                      |
|-----------------------|--------------------------------------------------|
| `journal_text`        | User's written journal/reflection entry          |
| `ambience_type`       | Environment context (calm, noisy, etc.)          |
| `time_of_day`         | Morning / afternoon / night                      |
| `previous_day_mood`   | Mood from the previous day                       |
| `face_emotion_hint`   | Facial expression hint (if available)            |
| `reflection_quality`  | Quality of the reflection (deep, surface, etc.)  |

### Target Columns:

| Column           | Type           | Description                    |
|------------------|----------------|--------------------------------|
| `emotional_state`| Categorical    | Emotion label (happy/sad/angry)|
| `intensity`      | Numeric        | Intensity of the emotion       |

---

## 🔤 NLP Pipeline

```
Step 1 → Fill missing values with empty string
Step 2 → Combine all 6 features into one text string per row
Step 3 → TF-IDF Vectorization (max_features = 5000)
Step 4 → LabelEncoder on emotional_state (for classification target)
Step 5 → intensity column used directly (for regression target)
```

---

## 🧠 Model Details

### 1️⃣ RandomForestClassifier
- **Purpose**: Predicts emotion **label**
- `n_estimators = 100`
- Input: TF-IDF matrix
- Output: Encoded emotion class → decoded via `LabelEncoder`

### 2️⃣ RandomForestRegressor
- **Purpose**: Predicts emotion **intensity score**
- `n_estimators = 100`
- Input: TF-IDF matrix
- Output: Continuous numeric intensity value

---

## 🔧 Decision Engine

The Decision Engine maps `(emotion, intensity)` → `(suggested_action, timing)`:

| Emotion          | Intensity | Suggested Action              | Timing    |
|------------------|-----------|-------------------------------|-----------|
| sad / sadness    | > 3       | Talk to a friend immediately  | Now       |
| sad / sadness    | ≤ 3       | Listen to music               | Later     |
| angry / anger    | > 3       | Take a break                  | Now       |
| angry / anger    | ≤ 3       | Go for a walk                 | Soon      |
| happy / joy      | Any       | Keep going!                   | Anytime   |
| Others           | Any       | Stay mindful                  | Flexible  |

---

## 📁 Project Structure

```
emotion-journal-system/
│
├── data/
│   ├── Sample_arvyax_reflective_dataset.csv    # Training data
│   └── arvyax_test_inputs_120.csv              # Test data
│
├── Emotions_.py                                # Main script
│
├── output/
│   └── final_output.csv                        # Predictions output
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-journal-system.git
cd emotion-journal-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
scikit-learn
numpy
```

---

## 🚀 Usage

### Run Batch Prediction (on test dataset)
```bash
python Emotions_.py
```
Output saved to `final_output.csv` with columns:
- `predicted_emotion`
- `predicted_intensity`
- `suggested_action`
- `when_to_do`

### Real-Time User Input Mode

After batch prediction, the script automatically starts interactive mode:

```
Enter your feelings (type 'exit' to stop)

You: I feel really low and exhausted today

--- RESULT ---
Emotion     : sad
Intensity   : 3.85
Suggested Action: Talk to a friend immediately
When        : Now
----------------------
```

Type `exit` to quit.

---

## 📊 Sample Output

| journal_text (sample)         | predicted_emotion | predicted_intensity | suggested_action             | when_to_do |
|-------------------------------|-------------------|---------------------|------------------------------|------------|
| "Feeling really down today"   | sad               | 4.2                 | Talk to a friend immediately | Now        |
| "Had a great productive day!" | happy             | 3.8                 | Keep going!                  | Anytime    |
| "Got frustrated at work"      | angry             | 2.9                 | Go for a walk                | Soon       |

---

## 🔮 Future Improvements

- [ ] Add Streamlit UI for interactive journaling experience
- [ ] Expand emotion classes (fear, surprise, disgust, anxiety)
- [ ] Use BERT / transformer models for deeper text understanding
- [ ] Add more contextual inputs (sleep hours, weather, activity level)
- [ ] Deploy as a REST API using FastAPI
- [ ] Store user history for mood trend analysis over time
- [ ] Add multilingual support (Hindi, etc.)

---

## 👩‍💻 Author

**Anshu Choudhary**  
📧 anshukumari88260@gmail.com  
🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, give it a star on GitHub!
