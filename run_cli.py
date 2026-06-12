import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess import train_test_split, StandardScaler, MinMaxScaler
from src.knn import KNNClassifier
from src.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

def download_dataset():
    """Downloads the Kaggle/UCI Heart Disease dataset if not already present."""
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    filepath = "heart_disease.csv"
    
    if not os.path.exists(filepath):
        print(f"[*] Downloading Heart Disease dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"[+] Dataset saved to {filepath}")
        except Exception as e:
            print(f"[-] Error downloading dataset: {e}")
            raise e
    else:
        print(f"[+] Dataset already present locally: {filepath}")
    return filepath

def run_grid_search(X_train, X_test, y_train, y_test, max_k=25):
    """Performs grid search over K and distance metrics, returning results."""
    metrics = ['euclidean', 'manhattan', 'minkowski']
    results = []

    for metric in metrics:
        train_accs = []
        test_accs = []
        for k in range(1, max_k + 1):
            knn = KNNClassifier(k=k, metric=metric)
            knn.fit(X_train, y_train)
            
            # Predict
            train_preds = knn.predict(X_train)
            test_preds = knn.predict(X_test)
            
            # Calculate Accuracies
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            results.append({
                'metric': metric,
                'k': k,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            
        # Plot K-tuning curve for this metric
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_k + 1), train_accs, label='Train Accuracy', marker='o', color='#3b82f6', linewidth=2)
        plt.plot(range(1, max_k + 1), test_accs, label='Test Accuracy', marker='s', color='#ef4444', linewidth=2)
        plt.title(f'KNN Hyperparameter Tuning (Metric: {metric.capitalize()})', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Number of Neighbors (K)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(range(1, max_k + 1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True, facecolor='#f8fafc', edgecolor='none')
        plt.tight_layout()
        
        plot_path = f"knn_tuning_{metric}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[+] Saved tuning curve plot to {plot_path}")

    return pd.DataFrame(results)

def plot_confusion_matrix_heatmap(cm, classes, title='Confusion Matrix', filename='confusion_matrix.png'):
    """Plots a clean, modern confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    
    # Annotate text
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight='bold')
                     
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[+] Saved confusion matrix plot to {filename}")

def main():
    print("="*60)
    print("      KNN Classifier From Scratch - Evaluation Pipeline      ")
    print("="*60)
    
    # 1. Download & Load Data
    filepath = download_dataset()
    df = pd.read_csv(filepath)
    
    print(f"\n[+] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Target Distribution (Heart Disease vs Normal):")
    print(df['target'].value_counts(normalize=True).to_string())
    
    # Split into features and label
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    # 2. Split dataset into Train and Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n[+] Data Split: Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # 3. Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[+] Features standardized (Mean = 0, Std = 1)")
    
    # 4. Hyperparameter tuning (Grid Search)
    print("\n[*] Running hyperparameter grid search...")
    grid_results = run_grid_search(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find the best combination
    best_row = grid_results.loc[grid_results['test_accuracy'].idxmax()]
    best_k = int(best_row['k'])
    best_metric = best_row['metric']
    best_acc = best_row['test_accuracy']
    
    print("\n" + "="*50)
    print("                   BEST PARAMETERS                  ")
    print("="*50)
    print(f"Optimal K: {best_k}")
    print(f"Optimal Distance Metric: {best_metric.capitalize()}")
    print(f"Test Accuracy: {best_acc:.4%}")
    print("="*50)
    
    # 5. Evaluate Best Model
    best_knn = KNNClassifier(k=best_k, metric=best_metric)
    best_knn.fit(X_train_scaled, y_train)
    
    test_preds = best_knn.predict(X_test_scaled)
    
    # Calculate evaluation metrics using custom functions
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds)
    
    print("\n[+] Classification Report (Custom Implementation):")
    print(classification_report(y_test, test_preds))
    
    # Save the best confusion matrix
    plot_confusion_matrix_heatmap(cm, classes=['Normal (0)', 'Disease (1)'], title=f'Confusion Matrix (K={best_k}, {best_metric.capitalize()})')
    
    # 6. Verify against Scikit-Learn KNeighborsClassifier
    print("\n" + "="*50)
    print("          COMPARISON WITH SCIKIT-LEARN             ")
    print("="*50)
    
    sklearn_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
    sklearn_knn.fit(X_train_scaled, y_train)
    sklearn_preds = sklearn_knn.predict(X_test_scaled)
    
    sklearn_acc = accuracy_score(y_test, sklearn_preds)
    
    print(f"Custom KNN Test Accuracy:  {acc:.6f}")
    print(f"sklearn KNN Test Accuracy: {sklearn_acc:.6f}")
    
    # Check if predictions are identical
    mismatch = np.sum(test_preds != sklearn_preds)
    if mismatch == 0:
        print("[SUCCESS] Success! Custom KNN predictions match scikit-learn predictions exactly.")
    else:
        print(f"[!] Warning: There are {mismatch} prediction mismatches. This is typically due to differing tie-breaking rules.")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
