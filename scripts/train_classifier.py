# train_classifier.py
# This script trains an XGBoost Classifier using SMOTE to handle the imbalanced dataset.

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def train_flood_classifier(filepath="labeled_flood_data.csv"):
    """
    Trains an XGBoost classifier using SMOTE to create a more balanced
    training set, leading to better performance on rare events.
    """
    print(f"Loading labeled data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        print("Please run the 'label_data_with_proxies.py' script first.")
        return

    # --- 1. Prepare Features and Target ---
    y = df['flood_event']
    X = df.drop(columns=['flood_event'])

    if 'moon_phase' in X.columns:
        X = pd.get_dummies(X, columns=['moon_phase'], prefix='phase', drop_first=True)

    X.columns = [str(col) for col in X.columns]

    # --- 2. Split Data First ---
    # It is crucial to split the data BEFORE applying SMOTE to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    
    print(f"Original training data size: {len(X_train)}")
    print(f"Flood events in original training set: {y_train.sum()}")

    # --- 3. Apply SMOTE to the Training Data ---
    print("\nApplying SMOTE to balance the training data...")
    
    # CORRECTED SECTION: Dynamically adjust SMOTE neighbors
    # SMOTE's k_neighbors must be less than the number of samples in the minority class.
    n_minority_samples = y_train.sum()
    if n_minority_samples < 2:
        print("❌ Error: Not enough flood event samples in the training set to apply SMOTE.")
        return
        
    # Set k_neighbors to be one less than the number of minority samples,
    # but no more than the default of 5, which is generally a good value.
    k_neighbors = min(n_minority_samples - 1, 5)
    print(f"Adjusting SMOTE k_neighbors to: {k_neighbors}")

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled training data size: {len(X_train_resampled)}")
    print(f"Flood events in resampled training set: {y_train_resampled.sum()}")

    # --- 4. Train XGBoost Classifier on Resampled Data ---
    print("\nTraining XGBoost Classifier on balanced data...")
    # We no longer need scale_pos_weight because the data is now balanced.
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    # --- 5. Save the Model ---
    model_filename = "floodsense_xgb_classifier.pkl"
    joblib.dump(model, model_filename)
    print(f"\n✅ Model trained and saved as '{model_filename}'")

    # --- 6. Evaluate the Model on the Original, Unseen Test Set ---
    print("\n--- Model Performance on Test Set ---")
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (after SMOTE)')
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    train_flood_classifier()
