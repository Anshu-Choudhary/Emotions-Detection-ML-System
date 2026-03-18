# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# ==============================
# 2. Load Data
# ==============================
train = pd.read_csv("Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv")
test = pd.read_csv("arvyax_test_inputs_120.xlsx - Sheet1.csv")

# ==============================
# 3. Preprocessing
# ==============================

# Fill missing values
train.fillna("", inplace=True)
test.fillna("", inplace=True)

# Combine text + useful features
def combine_features(df):
    return (
        df["journal_text"] + " " +
        df["ambience_type"] + " " +
        df["time_of_day"] + " " +
        df["previous_day_mood"] + " " +
        df["face_emotion_hint"] + " " +
        df["reflection_quality"]
    )

train["combined"] = combine_features(train)
test["combined"] = combine_features(test)

# ==============================
# 4. Text Vectorization
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train["combined"])
X_test = vectorizer.transform(test["combined"])

# ==============================
# 5. Encode Target (Emotion)
# ==============================
le = LabelEncoder()
y_emotion = le.fit_transform(train["emotional_state"])

# Intensity (numeric already)
y_intensity = train["intensity"]

# ==============================
# 6. Train Models
# ==============================

# Emotion Model (Classification)
emotion_model = RandomForestClassifier(n_estimators=100)
emotion_model.fit(X, y_emotion)

# Intensity Model (Regression)
intensity_model = RandomForestRegressor(n_estimators=100)
intensity_model.fit(X, y_intensity)

# ==============================
# 7. Predictions
# ==============================

pred_emotion = emotion_model.predict(X_test)
pred_intensity = intensity_model.predict(X_test)

# Decode emotion labels
pred_emotion = le.inverse_transform(pred_emotion)

# ==============================
# 8. Decision Engine (IMPORTANT)
# ==============================

def decide_action(emotion, intensity):
    
    if emotion in ["sad", "sadness", "low"]:
        if intensity > 3:
            return "Talk to a friend immediately", "Now"
        else:
            return "Listen to music", "Later"
    
    elif emotion in ["angry", "anger"]:
        if intensity > 3:
            return "Take a break", "Now"
        else:
            return "Go for a walk", "Soon"
    
    elif emotion in ["happy", "joy"]:
        return "Keep going!", "Anytime"
    
    else:
        return "Stay mindful", "Flexible"

# Apply decisions
actions = []
timings = []

for e, i in zip(pred_emotion, pred_intensity):
    act, time = decide_action(e, i)
    actions.append(act)
    timings.append(time)

# ==============================
# 9. Final Output
# ==============================

output = test.copy()
output["predicted_emotion"] = pred_emotion
output["predicted_intensity"] = pred_intensity
output["suggested_action"] = actions
output["when_to_do"] = timings

# Save file
output.to_csv("final_output.csv", index=False)

print(" Done! File saved as final_output.csv")
# ==============================
# 10. USER INPUT SYSTEM
# ==============================

def predict_from_user_input():

    print("\n Enter your feelings (type 'exit' to stop)\n")
    
    while True:
        user_text = input("You: ")
        
        if user_text.lower() == "exit":
            print(" Exiting...")
            break
        
        # Default values for other features
        ambience = "unknown"
        time_of_day = "unknown"
        prev_mood = "neutral"
        face_hint = "neutral"
        reflection = "normal"
        
        # Combine features (same as training)
        combined = (
            user_text + " " +
            ambience + " " +
            time_of_day + " " +
            prev_mood + " " +
            face_hint + " " +
            reflection
        )
        
        # Transform using trained vectorizer
        X_user = vectorizer.transform([combined])
        
        # Predict
        pred_em = emotion_model.predict(X_user)
        pred_em = le.inverse_transform(pred_em)[0]
        
        pred_int = intensity_model.predict(X_user)[0]
        
        # Decision
        action, timing = decide_action(pred_em, pred_int)
        
        # Output
        print("\n--- RESULT ---")
        print("Emotion:", pred_em)
        print("Intensity:", round(pred_int, 2))
        print("Suggested Action:", action)
        print("When:", timing)
        print("----------------------\n")


# Run it
predict_from_user_input()