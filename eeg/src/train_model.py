import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load Features (Assuming generated from previous script)
# X = np.load('X_features.npy')
# y = np.load('y_labels.npy')

# Placeholder for demo
X = np.random.rand(100, 196) # 196 features
y = np.random.randint(0, 4, 100)

# Split Data [cite: 668]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Models
models = {
    "SVM": SVC(kernel='rbf', C=10, gamma='scale'),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5) # Best performer per report
}

best_acc = 0
best_model = None

print("Training Models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\nBest Model: {best_model} with {best_acc:.4f} accuracy")

# Save Model and Scaler for Offline Prediction
joblib.dump(best_model, 'gesture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved as gesture_model.pkl")