import tensorflow as tf
from tensorflow.keras import layers, models

# ── Load & Preprocess ─────────────────────────────────────────────────
dataset_path = r"E:\Ai\Chessman-image-dataset\Chess"
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(28,28),
    batch_size=1
)
images = []
labels = []
for x, y in dataset:
    images.append(x[0].numpy())
    labels.append(y.numpy()[0])
X_train = tf.convert_to_tensor(images) / 255.0
y_train = tf.convert_to_tensor(labels)

# ── Build CNN ────────────────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(6, activation='softmax')
])

# ── Compile & Train ───────────────────────────────────────────────────
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# ── Evaluate ──────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_train, y_train)
print(f"Accuracy: {acc:.4f}")

model.save("chess_cnn_model.h5")



