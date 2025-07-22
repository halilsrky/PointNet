import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ayarlar
DATA_DIR = 'NPY'
BASE_NAME = 'TestRay{}'
INDEX = 236  # Görselleştirilecek dosya indexi (kullanıcı değiştirebilir)

# Dosya isimleri
base_file = BASE_NAME.format(INDEX) + '.npy'
rot_file = BASE_NAME.format(INDEX) + '_rot.npy'
scale_file = BASE_NAME.format(INDEX) + '_scale.npy'
trans_file = BASE_NAME.format(INDEX) + '_trans.npy'

file_list = [base_file, rot_file, scale_file, trans_file]
titles = ['Orijinal', 'Rotation', 'Scaling', 'Translation']

fig = plt.figure(figsize=(16, 4))
for i, (fname, title) in enumerate(zip(file_list, titles)):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"Dosya yok: {path}")
        continue
    data = np.load(path)
    points = data[:, :3]
    labels = data[:, 3]
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='coolwarm', s=2)
    ax.set_title(title)
    ax.set_axis_off()

plt.tight_layout()
plt.show()
