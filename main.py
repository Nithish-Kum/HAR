import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------
# 1. Load Raw Inertial Sensor Signals (128x6)
# ------------------------------------------------
def load_signals(folder, dataset_type):
    signals = []
    signal_types = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_"
    ]
    
    for signal in signal_types:
        data = np.loadtxt(folder + signal + dataset_type + ".txt")
        signals.append(data)
    
    return np.transpose(signals, (1, 2, 0))


# Load training data
X_train = load_signals(
    "UCI HAR Dataset/train/Inertial Signals/", "train"
)
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt")

# Load testing data
X_test = load_signals(
    "UCI HAR Dataset/test/Inertial Signals/", "test"
)
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt")

# Convert labels from 1–6 to 0–5
y_train = y_train - 1
y_test = y_test - 1

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ------------------------------------------------
# 2. Build Improved CNN-LSTM Model
# ------------------------------------------------
model = Sequential()

# First CNN Block
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(128, 6)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Second CNN Block
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# LSTM Layer
model.add(LSTM(128))

# Dense Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------
# 3. Train Model
# ------------------------------------------------
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ------------------------------------------------
# 4. Evaluate Model
# ------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("\n==============================")
print("Final Test Accuracy:", round(accuracy * 100, 2), "%")
print("==============================")

# Predict probabilities
y_pred = model.predict(X_test)

# Convert probabilities to class index
y_pred_classes = np.argmax(y_pred, axis=1)

# Activity label mapping
activity_labels = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

print("\n==============================")
print("Sample Predictions (First 10)")
print("==============================")

for i in range(10):
    print(f"Sample {i+1}")
    print("Predicted Activity :", activity_labels[y_pred_classes[i]])
    print("Actual Activity    :", activity_labels[int(y_test[i])])
    print("Confidence         :", round(np.max(y_pred[i]) * 100, 2), "%")
    print("----------------------------------")

print("\n==============================")
print("Classification Report")
print("==============================")
print(classification_report(y_test, y_pred_classes))

print("\n==============================")
print("Confusion Matrix")
print("==============================")
print(confusion_matrix(y_test, y_pred_classes))
# ------------------------------------------------
# 5. Save Model (Keras format recommended)
# ------------------------------------------------
model.save("har_model.keras")
print("\nModel saved as har_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Fix for LSTM conversion
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("har_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully")