import wfdb
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# -------------------------------
# BALANCING FUNCTION
# -------------------------------
def oversample_data(X, y):
    counter = Counter(y)
    max_count = max(counter.values())

    X_new, y_new = [], []

    for label in counter:
        idx = np.where(y == label)[0]
        repeat_times = max_count // len(idx)

        X_rep = np.tile(X[idx], (repeat_times, 1))
        y_rep = np.tile(y[idx], repeat_times)

        X_new.append(X_rep)
        y_new.append(y_rep)

    return np.vstack(X_new), np.hstack(y_new)

# -------------------------------
# STEP 1: LOAD TRAIN DATA
# -------------------------------
train_records = ['100', '101', '102', '103']

label_map = {'N': 0, 'V': 1, 'A': 2}
inv_label_map = {0: "Normal", 1: "Ventricular", 2: "Atrial"}

def build_dataset(records):
    X, y = [], []

    for rec in records:
        record = wfdb.rdrecord(f'data/{rec}')
        signal = record.p_signal[:, 0]

        ann = wfdb.rdann(f'data/{rec}', 'atr')
        beats = ann.sample
        labels = ann.symbol

        for i in range(len(beats)):
            if labels[i] not in label_map:
                continue

            start = beats[i] - 50
            end = beats[i] + 50

            if start > 0 and end < len(signal):
                X.append(signal[start:end])
                y.append(label_map[labels[i]])

    return np.array(X), np.array(y)

X, y = build_dataset(train_records)

print("Training samples:", len(X))
print("Class distribution:", Counter(y))

# -------------------------------
# STEP 2: TRAIN MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, y_train = oversample_data(X_train, y_train)

print("Balanced class distribution:", Counter(y_train))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Test Accuracy:", model.score(X_test, y_test))

# -------------------------------
# STEP 3: DYNAMIC PATIENT INPUT
# -------------------------------

files = os.listdir("data")
available_records = sorted(set(f.split('.')[0] for f in files if f.endswith('.dat')))

print("\nAvailable patients:", available_records)

user_input = input("\nEnter patient ID (or type ALL): ")

if user_input == "ALL":
    test_records = available_records
else:
    if user_input not in available_records:
        print("Invalid ID. Using default 104.")
        test_records = ['104']
    else:
        test_records = [user_input]

# -------------------------------
# STEP 4: LOOP THROUGH PATIENTS
# -------------------------------

for test_record in test_records:

    print(f"\n==============================")
    print(f"Testing Patient: {test_record}")
    print("==============================")

    record = wfdb.rdrecord(f'data/{test_record}')
    signal = record.p_signal[:, 0]

    ann = wfdb.rdann(f'data/{test_record}', 'atr')
    beats = ann.sample
    labels = ann.symbol

    history, y_true, y_pred = [], [], []

    print("\n--- MONITORING ---")

    count = 0

    for i in range(len(beats)):
        if labels[i] not in label_map:
            continue

        start = beats[i] - 50
        end = beats[i] + 50

        if start > 0 and end < len(signal):
            segment = signal[start:end]

            pred_class = model.predict(segment.reshape(1, -1))[0]
            actual_class = label_map[labels[i]]

            y_pred.append(pred_class)
            y_true.append(actual_class)

            pred = inv_label_map[pred_class]
            actual = inv_label_map[actual_class]

            history.append("Abnormal" if pred_class != 0 else "Normal")

            count += 1
            print(f"Beat {count}: Predicted = {pred} | Actual = {actual}")

            if count == 50:
                break

    # -------------------------------
    # VALIDATION
    # -------------------------------
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print("\nValidation Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0,1,2],
        target_names=["Normal","Ventricular","Atrial"],
        zero_division=0
    ))

    # -------------------------------
    # RISK
    # -------------------------------
    def analyze_trend(history):
        ratio = history.count("Abnormal") / len(history)
        if ratio > 0.4:
            return "High Risk"
        elif ratio > 0.1:
            return "Moderate Risk"
        return "Stable"

    trend = analyze_trend(history)
    print("\nFinal Risk:", trend)

    # -------------------------------
    # HEART RATE
    # -------------------------------
    rr = np.diff(beats) / 360
    hr = 60 / np.mean(rr)
    print("Heart Rate:", round(hr,2), "BPM")

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    start_index = None

    for i in range(len(beats)):
        if labels[i] in label_map:
            start_index = max(0, beats[i] - 200)
            break

    end_index = start_index + 5000

    plt.figure(figsize=(12,5))
    plt.plot(signal[start_index:end_index], label="ECG")

    for i in range(len(beats)):
        if beats[i] < start_index or beats[i] > end_index:
            continue

        if labels[i] not in label_map:
            continue

        x = beats[i] - start_index
        y_val = signal[beats[i]]

        if labels[i] == 'N':
            plt.scatter(x, y_val, color='green', s=40)
        elif labels[i] == 'V':
            plt.scatter(x, y_val, color='orange', s=120)
        else:
            plt.scatter(x, y_val, color='red', s=120)

    plt.title(f"ECG Visualization - Patient {test_record}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # -------------------------------
    # ALERT
    # -------------------------------
    if trend == "High Risk":
        print("⚠️ ALERT: High cardiac risk detected!")
