import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ Step 1: Load Data from CSV Files
data_path = "C:/Exports"  # Change this if needed
all_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

dfs = {}
for file in all_files:
    file_path = os.path.join(data_path, file)
    try:
        df = pd.read_csv(file_path, low_memory=False)  # Read CSV files
        if df.shape[0] == 0 or df.shape[1] == 0:
            print(f"❌ Skipping {file}: No valid data")
            continue
        dfs[file.replace(".csv", "")] = df
        print(f"🔹 {file} - Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading {file}: {e}")

# ✅ Step 2: Merge or Select a Relevant Dataset
# If multiple tables exist, use a main one (e.g., "HouseOwner.csv")
if "HouseOwner" in dfs:
    df = dfs["HouseOwner"]
else:
    print("⚠ No primary dataset found. Using first available dataset.")
    df = next(iter(dfs.values()))

# ✅ Step 3: Preprocessing
# Handling Missing Values
df.replace(["", " "], np.nan, inplace=True)  # Convert empty strings to NaN
df.fillna(method="ffill", inplace=True)  # Forward-fill missing values

# Drop columns with excessive missing values (less than 50% valid data kept)
df.dropna(axis=1, thresh=0.5 * len(df), inplace=True)

# Drop rows with excessive missing values (keep at least 70% valid data)
df.dropna(axis=0, thresh=0.7 * len(df.columns), inplace=True)

# ✅ Step 4: Feature Selection & Target Variable
# Convert categorical columns to numerical if needed
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category").cat.codes

# Define Features & Target Variable
target_column = "UpdatedBy"  # Change this if needed
if target_column not in df.columns:
    print(f"⚠ Target column '{target_column}' not found. Using first column as target.")
    target_column = df.columns[-1]  # Use last column as a fallback

X = df.drop(columns=[target_column])
y = df[target_column]

# Verify Data Integrity Before Splitting
if X.shape[0] == 0:
    raise ValueError("🚨 ERROR: Dataset is empty after preprocessing! Adjust data cleaning steps.")

print(f"✅ Data ready for training: Features {X.shape}, Target {y.shape}")

# ✅ Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 7: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")

# ✅ Step 8: Save Model for Future Use
import joblib
model_path = "C:/Exports/trained_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

