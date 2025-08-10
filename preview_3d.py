import os
import numpy as np
import pyvista as pv

def generate_3d_preview(hgt_path, out_dir='previews'):
    fname = os.path.basename(hgt_path)
    preview_path = os.path.join(out_dir, fname.replace('.hgt', '_3d.png'))
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
    skip = 5
    Z = Z[::skip, ::skip]
    nx, ny = Z.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    surface = pv.StructuredGrid(X, Y, Z)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(surface, cmap='terrain', show_edges=False, lighting=True)
    plotter.set_scale(zscale=1.0)
    plotter.camera_position = [
        (nx, ny, 800),
        (nx / 2, ny / 2, 200),
        (0, 0, 1)
    ]
    plotter.show(screenshot=preview_path)
    return preview_path
