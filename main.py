import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

data_path = "C:/Exports"  
all_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

dfs = {}
for file in all_files:
    file_path = os.path.join(data_path, file)
    try:
        df = pd.read_csv(file_path, low_memory=False)  
        if df.shape[0] == 0 or df.shape[1] == 0:
            print(f"❌ Skipping {file}: No valid data")
            continue
        dfs[file.replace(".csv", "")] = df
        print(f"🔹 {file} - Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading {file}: {e}")


if "HouseOwner" in dfs:
    df = dfs["HouseOwner"]
else:
    print("⚠ No primary dataset found. Using first available dataset.")
    df = next(iter(dfs.values()))


df.replace(["", " "], np.nan, inplace=True) 
df.fillna(method="ffill", inplace=True)  


df.dropna(axis=1, thresh=0.5 * len(df), inplace=True)


df.dropna(axis=0, thresh=0.7 * len(df.columns), inplace=True)


for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category").cat.codes


target_column = "UpdatedBy"  
if target_column not in df.columns:
    print(f"⚠ Target column '{target_column}' not found. Using first column as target.")
    target_column = df.columns[-1]  

X = df.drop(columns=[target_column])
y = df[target_column]


if X.shape[0] == 0:
    raise ValueError("🚨 ERROR: Dataset is empty after preprocessing! Adjust data cleaning steps.")

print(f"✅ Data ready for training: Features {X.shape}, Target {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")



scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Cross-validation scores:", scores)
print("Mean CV Score:", scores.mean())
import joblib
model_path = "C:/Exports/trained_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

