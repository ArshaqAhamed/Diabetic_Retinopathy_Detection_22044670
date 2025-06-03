import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
image_folder = r"D:\OneDrive - University of Hertfordshire\UH SLTTC\Final Year Project\Dataset\aptos2019-blindness-detection2\train_images"
labels_file = r"D:\OneDrive - University of Hertfordshire\UH SLTTC\Final Year Project\Dataset\aptos2019-blindness-detection2\train.csv"

# Parameters
img_size = (64, 64)

# Load CSV
data = pd.read_csv(labels_file)
inputs = []
targets = []

# Load and preprocess images
for i in range(len(data)):
    img_path = os.path.join(image_folder, f"{data['id_code'][i]}.png")
    img = Image.open(img_path).resize(img_size).convert('L')
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    inputs.append(img_array.flatten())
    
    label = 0 if data['diagnosis'][i] == 0 else 1
    targets.append(label)

X = np.array(inputs)
y = np.array(targets)

# Normalize using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

import joblib
joblib.dump(scaler, 'binary_scaler.pkl')

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model summary
print("Model Summary:")
print("----------------")
model.summary()

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop])

# Save model
model.save("dr_ann_model.h5")
print("\nModel saved as 'dr_ann_model.h5'")

# Evaluate
y_pred_probs = model.predict(X_test).flatten()
y_pred = np.round(y_pred_probs)

# Confusion matrix and AUC
conf_matrix = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_probs)

TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP + 1e-10)
recall = TP / (TP + FN + 1e-10)
f1_score = 2 * precision * recall / (precision + recall + 1e-10)

# Print metrics
print("\nModel Performance Metrics:")
print("---------------------------")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-Score : {f1_score:.2f}")
print(f"AUC      : {auc:.2f}")

# Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['No DR', 'DR'])
plt.yticks(tick_marks, ['No DR', 'DR'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.show()
