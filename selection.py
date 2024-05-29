import os
import shutil
import numpy as np

# Pfad zum Ordner mit den Bilddateien
path_X = 'Images'

train_path = 'Data/Training'
validate_path = 'Data/Validation'

# Ordner "Data" und darin die Unterordner "Training" und "Validation" erstellen
os.makedirs(train_path, exist_ok=True)
os.makedirs(validate_path, exist_ok=True)

# Alle Unterordner im Ordner "Images" auflisten
for folder in os.listdir(path_X):
    if os.path.isdir(os.path.join(path_X, folder)):
        
        # Erstellen Sie entsprechende Unterordner in "Training" und "Validierung"
        os.makedirs(f'Data/Training/{folder}', exist_ok=True)
        os.makedirs(f'Data/Validation/{folder}', exist_ok=True)

        # Alle Bilddateien in jedem Unterordner auflisten
        files = [f for f in os.listdir(os.path.join(path_X, folder)) if os.path.isfile(os.path.join(path_X, folder, f))]

        # Bilddateien aus der Liste mischen, um eine zufällige Auswahl zu gewährleisten
        np.random.shuffle(files)

        # Aufteilen der Bilddateien in 80% für das Training und 20% für die Validierung
        split_idx = int(0.8 * len(files))
        train_files = files[:split_idx]
        validation_files = files[split_idx:]

        # Entsprechenden Bilddateien in die entsprechenden Unterordner kopieren
        for file in train_files:
            shutil.copy(os.path.join(path_X, folder, file), f'Data/Training/{folder}')
        for file in validation_files:
            shutil.copy(os.path.join(path_X, folder, file), f'Data/Validation/{folder}')