# train_spatio_temporal_model.py
# This script trains the final XGBoost Classifier on the full spatio-temporal dataset.

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def train_spatio_temporal_classifier(filepath="spatio_temporal_flood_data.csv"):
    """
    Trains an XGBoost classifier on the combined spatio-temporal data.
    """
    print(f"Loading spatio-temporal data from '{filepath}'...")
    try:
        # The multi-index is helpful for analysis but needs to be reset for training.
        df = pd.read_csv(filepath, index_col=[0, 1], parse_dates=True)
        df.reset_index(inplace=True)
        df.rename(columns={'level_0': 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        print("Please run the 'create_final_dataset.py' script first.")
        return

    # --- 1. Prepare Features and Target ---
    y = df['flood_event']
    # Drop the target and any non-feature columns like ward_name
    X = df.drop(columns=['flood_event', 'ward_name'])

    # One-hot encode categorical features if they exist
    if 'moon_phase' in X.columns:
        X = pd.get_dummies(X, columns=['moon_phase'], prefix='phase', drop_first=True)

    X.columns = [str(col) for col in X.columns]

    # --- 2. Split Data using Stratified Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    
    print(f"Original training data size: {len(X_train)}")
    print(f"Flood events in original training set: {y_train.sum()}")

    # --- 3. Apply SMOTE to the Training Data ---
    print("\nApplying SMOTE to balance the training data...")
    # Adjust k_neighbors based on the number of flood events
    n_minority_samples = y_train.sum()
    k_neighbors = min(n_minority_samples - 1, 5)
    print(f"Using {k_neighbors} neighbors for SMOTE.")

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled training data size: {len(X_train_resampled)}")

    # --- 4. Train XGBoost Classifier ---
    print("\nTraining Spatio-Temporal XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=500, # Increased estimators for the more complex dataset
        learning_rate=0.05,
        max_depth=7, # Increased depth
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    # --- 5. Save the Final Model ---
    model_filename = "floodsense_spatio_temporal_classifier.pkl"
    joblib.dump(model, model_filename)
    print(f"\n✅ Final model trained and saved as '{model_filename}'")

    # --- 6. Evaluate the Model ---
    print("\n--- Final Model Performance on Test Set ---")
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
    plt.title('Spatio-Temporal Model Confusion Matrix')
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    train_spatio_temporal_classifier()
