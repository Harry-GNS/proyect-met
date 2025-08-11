import os
import numpy as np
import matplotlib.pyplot as plt

def generate_hgt_preview(hgt_path, out_dir='previews'):
    fname = os.path.basename(hgt_path)
    preview_path = os.path.join(out_dir, fname.replace('.hgt', '_preview.png'))
    if os.path.exists(preview_path):
        return preview_path
    with open(hgt_path, 'rb') as f:
        data = np.fromfile(f, dtype='>i2')
        size = int(np.sqrt(data.size))
        if size * size != data.size:
            print(f'Tamaño no cuadrado en {hgt_path}')
            return None
        Z = data.reshape((size, size)).astype(float)
        void_mask = Z <= -32000
        if void_mask.any():
            Z[void_mask] = np.nan
            Z = np.where(np.isnan(Z), np.nanmin(Z), Z)
    plt.figure(figsize=(3, 3))
    plt.imshow(Z, cmap='terrain', origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(preview_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return preview_path

def generate_all_previews(hgt_dir='data', out_dir='previews'):
    for dirpath, _, filenames in os.walk(hgt_dir):
        for fname in filenames:
            if fname.lower().endswith('.hgt'):
                hgt_path = os.path.join(dirpath, fname)
                generate_hgt_preview(hgt_path, out_dir)
