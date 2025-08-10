import pyvista as pv
import numpy as np
from scipy.ndimage import gaussian_filter

# Leer el archivo .hgt
file_path = 'S01W076.hgt'
size = 1201
with open(file_path, 'rb') as f:
    data = np.fromfile(f, dtype='>i2')
    heights = data.reshape((size, size)).astype(float)

# Suavizar
sigma = 20.0
Z = gaussian_filter(heights, sigma=sigma)

# Crear malla (opcional: reducir resolución)
skip = 5
Z = Z[::skip, ::skip]
nx, ny = Z.shape
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Crear modelo sólido para impresión 3D sin bordes abruptos
min_height = np.min(Z)
base_thickness = 50  # Grosor de la base en unidades de altura
base_height = min_height - base_thickness

# Suavizar los bordes para evitar paredes verticales
edge_fade = 10  # Píxeles desde el borde para suavizar
Z_smooth = Z.copy()

# Aplicar degradado en los bordes
for i in range(edge_fade):
    # Borde superior
    Z_smooth[i, :] = np.minimum(Z_smooth[i, :], base_height + (Z_smooth[i, :] - base_height) * (i / edge_fade))
    # Borde inferior  
    Z_smooth[-(i+1), :] = np.minimum(Z_smooth[-(i+1), :], base_height + (Z_smooth[-(i+1), :] - base_height) * (i / edge_fade))
    # Borde izquierdo
    Z_smooth[:, i] = np.minimum(Z_smooth[:, i], base_height + (Z_smooth[:, i] - base_height) * (i / edge_fade))
    # Borde derecho
    Z_smooth[:, -(i+1)] = np.minimum(Z_smooth[:, -(i+1)], base_height + (Z_smooth[:, -(i+1)] - base_height) * (i / edge_fade))

# Crear superficie superior suavizada
surface_top = pv.StructuredGrid(X, Y, Z_smooth)

# Crear base
base_mesh = pv.StructuredGrid(X, Y, np.full_like(Z, base_height))

# Crear volumen sólido extrudiendo hacia abajo
points_all = []
faces_all = []

# Agregar puntos de la superficie superior
for i in range(nx):
    for j in range(ny):
        points_all.append([X[i, j], Y[i, j], Z_smooth[i, j]])

# Agregar puntos de la base
for i in range(nx):
    for j in range(ny):
        points_all.append([X[i, j], Y[i, j], base_height])

# Crear caras laterales conectando superficie con base
offset = nx * ny
for i in range(nx - 1):
    for j in range(ny - 1):
        # Cara superior (topografía)
        p1 = i * ny + j
        p2 = i * ny + (j + 1)
        p3 = (i + 1) * ny + (j + 1)
        p4 = (i + 1) * ny + j
        faces_all.extend([4, p1, p2, p3, p4])
        
        # Cara inferior (base)
        p1_base = p1 + offset
        p2_base = p2 + offset  
        p3_base = p3 + offset
        p4_base = p4 + offset
        faces_all.extend([4, p4_base, p3_base, p2_base, p1_base])  # Orden invertido para normal correcta

# Crear caras laterales en los bordes
# Solo en el perímetro exterior para cerrar el volumen
for i in range(nx - 1):
    # Borde frontal (j=0)
    p1_top = i * ny + 0
    p2_top = (i + 1) * ny + 0
    p1_bot = p1_top + offset
    p2_bot = p2_top + offset
    faces_all.extend([4, p1_top, p1_bot, p2_bot, p2_top])
    
    # Borde trasero (j=ny-1)
    p1_top = i * ny + (ny - 1)
    p2_top = (i + 1) * ny + (ny - 1)
    p1_bot = p1_top + offset
    p2_bot = p2_top + offset
    faces_all.extend([4, p2_top, p2_bot, p1_bot, p1_top])

for j in range(ny - 1):
    # Borde izquierdo (i=0)
    p1_top = 0 * ny + j
    p2_top = 0 * ny + (j + 1)
    p1_bot = p1_top + offset
    p2_bot = p2_top + offset
    faces_all.extend([4, p1_top, p2_top, p2_bot, p1_bot])
    
    # Borde derecho (i=nx-1)
    p1_top = (nx - 1) * ny + j
    p2_top = (nx - 1) * ny + (j + 1)
    p1_bot = p1_top + offset
    p2_bot = p2_top + offset
    faces_all.extend([4, p2_top, p1_top, p1_bot, p2_bot])

# Crear malla sólida completa
solid_mesh = pv.PolyData(points_all, faces_all)

# Visualizar modelo sólido sin bordes abruptos
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

# Guardar modelo para impresión 3D
# Exportar como STL para impresión 3D
try:
    solid_mesh.save('terreno_3d_solido.stl')
    print("Modelo guardado como 'terreno_3d_solido.stl' para impresión 3D")
except:
    print("Error al guardar el archivo STL")

# Mostrar información del modelo
print(f"Dimensiones del modelo:")
print(f"X: {np.max(X) - np.min(X):.1f} unidades")
print(f"Y: {np.max(Y) - np.min(Y):.1f} unidades") 
print(f"Z: {np.max(Z_smooth) - base_height:.1f} unidades")
print(f"Grosor de base: {base_thickness} unidades")
print(f"Suavizado de bordes: {edge_fade} píxeles")
