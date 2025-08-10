import pyvista as pv
import numpy as np
from scipy.ndimage import gaussian_filter

def process_hgt(file_path, stl_output='terreno_3d_solido.stl'):
    size = 1201
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype='>i2')
        heights = data.reshape((size, size)).astype(float)

    sigma = 20.0
    Z = gaussian_filter(heights, sigma=sigma)
    skip = 5
    Z = Z[::skip, ::skip]
    nx, ny = Z.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    min_height = np.min(Z)
    base_thickness = 50
    base_height = min_height - base_thickness
    edge_fade = 10
    Z_smooth = Z.copy()
    for i in range(edge_fade):
        Z_smooth[i, :] = np.minimum(Z_smooth[i, :], base_height + (Z_smooth[i, :] - base_height) * (i / edge_fade))
        Z_smooth[-(i+1), :] = np.minimum(Z_smooth[-(i+1), :], base_height + (Z_smooth[-(i+1), :] - base_height) * (i / edge_fade))
        Z_smooth[:, i] = np.minimum(Z_smooth[:, i], base_height + (Z_smooth[:, i] - base_height) * (i / edge_fade))
        Z_smooth[:, -(i+1)] = np.minimum(Z_smooth[:, -(i+1)], base_height + (Z_smooth[:, -(i+1)] - base_height) * (i / edge_fade))

    surface_top = pv.StructuredGrid(X, Y, Z_smooth)
    base_mesh = pv.StructuredGrid(X, Y, np.full_like(Z, base_height))
    points_all = []
    faces_all = []
    for i in range(nx):
        for j in range(ny):
            points_all.append([X[i, j], Y[i, j], Z_smooth[i, j]])
    for i in range(nx):
        for j in range(ny):
            points_all.append([X[i, j], Y[i, j], base_height])
    offset = nx * ny
    for i in range(nx - 1):
        for j in range(ny - 1):
            p1 = i * ny + j
            p2 = i * ny + (j + 1)
            p3 = (i + 1) * ny + (j + 1)
            p4 = (i + 1) * ny + j
            faces_all.extend([4, p1, p2, p3, p4])
            p1_base = p1 + offset
            p2_base = p2 + offset  
            p3_base = p3 + offset
            p4_base = p4 + offset
            faces_all.extend([4, p4_base, p3_base, p2_base, p1_base])
    for i in range(nx - 1):
        p1_top = i * ny + 0
        p2_top = (i + 1) * ny + 0
        p1_bot = p1_top + offset
        p2_bot = p2_top + offset
        faces_all.extend([4, p1_top, p1_bot, p2_bot, p2_top])
        p1_top = i * ny + (ny - 1)
        p2_top = (i + 1) * ny + (ny - 1)
        p1_bot = p1_top + offset
        p2_bot = p2_top + offset
        faces_all.extend([4, p2_top, p2_bot, p1_bot, p1_top])
    for j in range(ny - 1):
        p1_top = 0 * ny + j
        p2_top = 0 * ny + (j + 1)
        p1_bot = p1_top + offset
        p2_bot = p2_top + offset
        faces_all.extend([4, p1_top, p2_top, p2_bot, p1_bot])
        p1_top = (nx - 1) * ny + j
        p2_top = (nx - 1) * ny + (j + 1)
        p1_bot = p1_top + offset
        p2_bot = p2_top + offset
        faces_all.extend([4, p2_top, p1_top, p1_bot, p2_bot])
    solid_mesh = pv.PolyData(points_all, faces_all)
    plotter = pv.Plotter()
    plotter.add_mesh(solid_mesh, cmap='terrain', show_edges=False, lighting=True)
    plotter.enable_eye_dome_lighting()
    plotter.show_grid()
    plotter.set_scale(zscale=1.0)
    plotter.camera_position = [
        (nx, ny, 800),
        (nx / 2, ny / 2, 200),
        (0, 0, 1)
    ]
    plotter.show(title='Terreno 3D Sólido para Impresión')
    try:
        solid_mesh.save(stl_output)
        print(f"Modelo guardado como '{stl_output}' para impresión 3D")
    except:
        print("Error al guardar el archivo STL")
    print(f"Dimensiones del modelo:")
    print(f"X: {np.max(X) - np.min(X):.1f} unidades")
    print(f"Y: {np.max(Y) - np.min(Y):.1f} unidades") 
    print(f"Z: {np.max(Z_smooth) - base_height:.1f} unidades")
    print(f"Grosor de base: {base_thickness} unidades")
    print(f"Suavizado de bordes: {edge_fade} píxeles")
