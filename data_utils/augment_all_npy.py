import os
import numpy as np

np.random.seed(42)

# Augmentation fonksiyonları
def random_rotation(points):
    theta = np.random.uniform(0, 2 * np.pi)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return points @ rot_matrix.T

def random_scaling(points, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return points * scale

def random_translation(points, shift_range=0.2):
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    return points + shifts

# Klasör ve dosya ayarları
data_dir = 'NPY'
files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and f.startswith('TestRay')]

for file in files:
    data = np.load(os.path.join(data_dir, file))  # Nx4: X Y Z label
    points = data[:, :3]
    labels = data[:, 3:]

    # 1. Rotation
    rot_points = random_rotation(points)
    np.save(os.path.join(data_dir, file.replace('.npy', '_rot.npy')), np.hstack([rot_points, labels]))

    # 2. Scaling
    scale_points = random_scaling(points)
    np.save(os.path.join(data_dir, file.replace('.npy', '_scale.npy')), np.hstack([scale_points, labels]))

    # 3. Translation
    trans_points = random_translation(points)
    np.save(os.path.join(data_dir, file.replace('.npy', '_trans.npy')), np.hstack([trans_points, labels]))

    # 4. Orijinal dosya da kalsın
    print(f"Augmented: {file}")

print("Tüm augmentasyonlar tamamlandı.")
