import os
import pathlib
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def rename_images_in_directory(directory, prefix="img"):
    """
    Переименовывает все изображения в указанной директории и её подпапках,
    присваивая им последовательные имена в формате "image_номер.расширение".
    """
    if not os.path.isdir(directory):
        print(f"Ошибка: Папка '{directory}' не найдена.")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []

    # Проходим по всем файлам в указанной папке и её подпапках
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    # Сортируем файлы по имени для стабильного порядка номеров
    image_files.sort()

    # Переименовываем файлы
    for index, old_path in enumerate(image_files, start=1):
        directory, filename = os.path.split(old_path)
        ext = os.path.splitext(filename)[1]  # Расширение файла
        new_name = f"{prefix}_{index}{ext}"
        new_path = os.path.join(directory, new_name)

        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")


# Прямой URL для скачивания .zip файла
Type_of_eyes_url = (
    "https://drive.google.com/uc?export=download&id=1nsD3EqrTYl_GE2WO7dloJ6QSyNQGtzfq"
)

# Скачивание .zip файла
zip_path = tf.keras.utils.get_file('Type_of_eyes.zip', origin=Type_of_eyes_url, extract=False)
print(f"Zip file downloaded to: {zip_path}")

extracted_path = pathlib.Path(zip_path).parent / "Type_of_eyes"

if not extracted_path.is_dir():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:  # Распаковка .zip файла
        zip_ref.extractall(extracted_path)
    print(f"Files extracted to: {extracted_path}")
    rename_images_in_directory(extracted_path)  # Вызов функции для переименования файлов

# Проверка, что данные были успешно распакованы
Type_of_eyes_data = extracted_path
if not Type_of_eyes_data.exists():
    raise FileNotFoundError(f"Data not found at {Type_of_eyes_data}")

data_dir = pathlib.Path(f"{extracted_path}/Type_of_eyes")  # Каталог с изображениями, разделёнными по классам

# Размер изображений, к которому они будут приведены перед подачей в модель
img_height, img_width = 180, 180
batch_size = 32  # Размер батча (количество изображений, подаваемых в модель за раз)

# Загружаем изображения для обучения (80%) и валидации (20%)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=234,  # Фиксируем случайность для воспроизводимости
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=234,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Получаем список классов (они соответствуют именам папок внутри data_dir)
class_names = train_ds.class_names
print(f"Классы: {class_names}")

# Нормализация данных: приводим значения пикселей к диапазону [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Оптимизация загрузки данных (кэширование и предварительная загрузка в память)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Создаём сверточную нейронную сеть (CNN) для классификации изображений глаз
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),  # Преобразуем 2D-карту признаков в 1D-вектор
    tf.keras.layers.Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами
    tf.keras.layers.Dense(len(class_names), activation='softmax'),  # Выходной слой для предсказания классов
])

# Компилируем модель, выбирая оптимизатор, функцию ошибки и метрику качества
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Обучаем модель на тренировочных данных и оцениваем её на валидационных
epochs = 15  # Количество эпох (полных проходов по данным)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Оцениваем работу модели: собираем предсказания и реальные классы
y_true = []  # Истинные метки классов
y_pred = []  # Предсказанные моделью классы

for images, labels in val_ds:
    preds = model.predict(images)  # Получаем предсказания модели
    y_true.extend(labels.numpy())  # Добавляем истинные метки классов
    y_pred.extend(np.argmax(preds, axis=1))  # Преобразуем вероятности в конкретные классы

# Строим матрицу путанности для оценки качества классификации
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица путанности")
plt.show()

# Выводим отчёт по классификации: точность, полнота и F1-мера для каждого класса
print(classification_report(y_true, y_pred, target_names=class_names))
