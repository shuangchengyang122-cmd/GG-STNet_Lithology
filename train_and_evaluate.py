import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gg_stnet_model import build_gg_stnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_preprocess_data(csv_path, window_size=64, step_size=2):
    print(f"[*] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    features = ['GR', 'AC', 'DEN', 'RT', 'SP']
    
    # Z-score normalization and type casting for GPU compatibility
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].values).astype(np.float32)
    labels = df['Lithology_Label'].values.astype(np.int32)
    
    # Construct sliding window tensors
    X, y = [], []
    for i in range(0, len(scaled_features) - window_size, step_size):
        X.append(scaled_features[i : i + window_size])
        y.append(labels[i + window_size - 1]) 
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def mc_dropout_predict(model, X_test, num_samples=50):
    print(f"\n[*] Running MC Dropout Uncertainty Quantification ({num_samples} passes)...")
    predictions = []
    
    # Execute multiple stochastic forward passes
    for i in range(num_samples):
        preds = model.predict(X_test, verbose=0)
        predictions.append(preds)
        if (i + 1) % 10 == 0:
            print(f"    - Completed {i + 1}/{num_samples} passes...")
            
    predictions = np.array(predictions)
    expected_prob = np.mean(predictions, axis=0)
    uncertainty_variance = np.var(predictions, axis=0)
    
    final_pred = np.argmax(expected_prob, axis=1)
    return final_pred, expected_prob, uncertainty_variance

if __name__ == '__main__':
    data_path = os.path.join("sample_data", "dummy_well_01.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Cannot find {data_path}! Run generate_dummy_data.py first.")

    X, y = load_and_preprocess_data(data_path, window_size=64, step_size=2)
    print(f"[*] Generated Tensor Shapes: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Train set: {X_train.shape[0]} windows | Test set: {X_test.shape[0]} windows")
    
    print("\n[*] Building GG-STNet Architecture...")
    model = build_gg_stnet(sequence_length=64, num_features=5, num_classes=7)
    
    print("\n[*] Starting Model Training...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate using MC Dropout
    final_pred, probs, variance = mc_dropout_predict(model, X_test, num_samples=50)
    
    print("\n" + "="*50)
    print("[SUCCESS] ALL PIPELINES EXECUTED FLAWLESSLY.")
    print("="*50)
    print(f"Sample Result for the First Test Window:")
    print(f" -> True Class: {y_test[0]}")
    print(f" -> Predicted Class (MC Mean): {final_pred[0]}")
    print(f" -> Uncertainty Variance Map (7 classes):\n {np.round(variance[0], 6)}")