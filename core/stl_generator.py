from typing import Callable, Optional
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import pyvista as pv
from core.paths import Paths
from core.hgt_utils import read_hgt_array, find_hgt_files

ProgressCB = Optional[Callable[[int, str], None]]

def generate_stl_from_hgt(hgt_path: str, out_dir: Optional[str] = None, progress: ProgressCB = None) -> str:
    def p(val, msg):
        if progress: progress(val, msg)

    Paths.ensure()
    if out_dir is None:
        out_dir = str(Paths.stl_dir)

    name = Path(hgt_path).stem
    out_path = str(Path(out_dir) / f"{name}_solido.stl")

    p(5, "Leyendo HGT...")
    Z = read_hgt_array(hgt_path)

    p(20, "Suavizando elevación...")
    Z = gaussian_filter(Z, sigma=5.0, truncate=2.5)

    p(30, "Submuestreando...")
    skip = 6
    Z = Z[::skip, ::skip]
    nx, ny = Z.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    p(40, "Construyendo base y bordes...")
    # Escalado vertical y normalización (similar a tu lógica)
    Z_scaled = Z * 0.12
    zmin, zmax = float(np.min(Z_scaled)), float(np.max(Z_scaled))
    rng = zmax - zmin if zmax > zmin else 1.0
    Z_norm = (Z_scaled - zmin) / rng
    compression_factor = 0.6
    Z_comp = np.where(Z_norm > 0.3, 0.3 + (Z_norm - 0.3)*compression_factor, Z_norm)
    Z_scaled = zmin + Z_comp * rng

    zmin = float(np.min(Z_scaled))
    base_thickness = 20.0
    base_height = zmin - base_thickness

    # Borde suave hacia la base
    edge_fade = 10
    Z_smooth = Z_scaled.copy()
    for i in range(edge_fade):
        alpha = i / edge_fade
        Z_smooth[i, :]       = np.minimum(Z_smooth[i, :],       base_height + (Z_smooth[i, :]       - base_height) * alpha)
        Z_smooth[-(i+1), :]  = np.minimum(Z_smooth[-(i+1), :],  base_height + (Z_smooth[-(i+1), :]  - base_height) * alpha)
        Z_smooth[:, i]       = np.minimum(Z_smooth[:, i],       base_height + (Z_smooth[:, i]       - base_height) * alpha)
        Z_smooth[:, -(i+1)]  = np.minimum(Z_smooth[:, -(i+1)],  base_height + (Z_smooth[:, -(i+1)]  - base_height) * alpha)

    p(55, "Construyendo malla sólida...")
    points = []
    # superficie
    for i in range(nx):
        for j in range(ny):
            points.append([X[i, j], Y[i, j], Z_smooth[i, j]])
    # base
    for i in range(nx):
        for j in range(ny):
            points.append([X[i, j], Y[i, j], base_height])

    offset = nx * ny
    faces = []
    # top y base
    for i in range(nx - 1):
        for j in range(ny - 1):
            p1 = i * ny + j
            p2 = i * ny + (j + 1)
            p3 = (i + 1) * ny + (j + 1)
            p4 = (i + 1) * ny + j
            faces.extend([4, p1, p2, p3, p4])
            p1b, p2b, p3b, p4b = p1 + offset, p2 + offset, p3 + offset, p4 + offset
            faces.extend([4, p4b, p3b, p2b, p1b])

    # lados
    for i in range(nx - 1):
        # izquierda (j=0)
        p1t = i * ny + 0
        p2t = (i + 1) * ny + 0
        p1b = p1t + offset
        p2b = p2t + offset
        faces.extend([4, p1t, p1b, p2b, p2t])
        # derecha (j=ny-1)
        p1t = i * ny + (ny - 1)
        p2t = (i + 1) * ny + (ny - 1)
        p1b = p1t + offset
        p2b = p2t + offset
        faces.extend([4, p2t, p2b, p1b, p1t])

    for j in range(ny - 1):
        # arriba (i=0)
        p1t = 0 * ny + j
        p2t = 0 * ny + (j + 1)
        p1b = p1t + offset
        p2b = p2t + offset
        faces.extend([4, p1t, p2t, p2b, p1b])
        # abajo (i=nx-1)
        p1t = (nx - 1) * ny + j
        p2t = (nx - 1) * ny + (j + 1)
        p1b = p1t + offset
        p2b = p2t + offset
        faces.extend([4, p2t, p1t, p1b, p2b])

    p(70, "Optimizando mesh...")
    points_array = np.array(points, dtype=np.float32)
    # convertir quads a triángulos para pyvista
    triangles = []
    for k in range(0, len(faces), 5):
        if k + 4 < len(faces):
            _, a, b, c, d = faces[k:k+5]
            triangles.extend([3, a, b, c])
            triangles.extend([3, a, c, d])
    mesh = pv.PolyData(points_array, np.array(triangles))
    mesh = mesh.clean(tolerance=1e-3)

    p(90, "Guardando STL...")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mesh.save(out_path)

    p(100, "Completado")
    return out_path

def generate_full_ecuador_map(out_dir: Optional[str] = None, progress_callback: ProgressCB = None) -> str:
    """
    Genera un modelo 3D combinado de todo el territorio de Ecuador utilizando
    todos los archivos HGT disponibles que intersectan con el país.

    Args:
        out_dir: Directorio de salida para el archivo STL resultante
        progress_callback: Función de callback para reportar progreso

    Returns:
        Ruta al archivo STL generado
    """
    def p(val, msg):
        if progress_callback:
            progress_callback(val, msg)

    # Aseguramos que existan las carpetas necesarias
    Paths.ensure()
    if out_dir is None:
        out_dir = str(Paths.stl_dir)

    # Nombre de archivo para el mapa completo
    out_path = str(Path(out_dir) / "ecuador_completo.stl")

    # 1. Encontrar todos los archivos HGT de Ecuador
    p(5, "Buscando archivos HGT de Ecuador...")
    hgt_files = find_hgt_files(only_ecuador=True)
    if not hgt_files:
        raise ValueError("No se encontraron archivos HGT para Ecuador")

    p(10, f"Se encontraron {len(hgt_files)} archivos HGT")

    # 2. Determinar las dimensiones del mapa completo
    min_lat = min(f["lat"] for f in hgt_files)
    max_lat = max(f["lat"] + 1 for f in hgt_files)
    min_lon = min(f["lon"] for f in hgt_files)
    max_lon = max(f["lon"] + 1 for f in hgt_files)

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    p(15, f"Área total: {lon_range}° x {lat_range}° ({min_lon}°-{max_lon}°E, {min_lat}°-{max_lat}°N)")

    # 3. Crear un grid combinado (submuestreado para manejar tamaño)
    skip = 12  # Factor de submuestreo
    points_per_degree = 1201 // skip  # SRTM tiene 1201 puntos por grado

    width = lon_range * points_per_degree
    height = lat_range * points_per_degree

    p(20, f"Creando grid combinado ({width}x{height} puntos)...")
    combined_dem = np.zeros((height, width), dtype=np.float32)
    combined_dem.fill(np.nan)  # Inicializar con NaN para áreas sin datos

    # 4. Cargar cada archivo HGT y combinarlo en el grid
    total_files = len(hgt_files)
    for i, hgt_file in enumerate(hgt_files):
        prog_val = 20 + int((i / total_files) * 40)  # Progreso de 20% a 60%
        p(prog_val, f"Procesando {hgt_file['name']} ({i+1}/{total_files})...")

        try:
            # Cargar y submuestrear el archivo HGT
            dem = read_hgt_array(hgt_file["path"])
            dem = dem[::skip, ::skip]

            # Calcular la posición en el grid combinado
            x_offset = (hgt_file["lon"] - min_lon) * points_per_degree
            y_offset = (max_lat - hgt_file["lat"] - 1) * points_per_degree

            x_offset = int(x_offset)
            y_offset = int(y_offset)

            # Insertar en el grid combinado
            h, w = dem.shape
            combined_dem[y_offset:y_offset+h, x_offset:x_offset+w] = dem

        except Exception as e:
            p(prog_val, f"Error en {hgt_file['name']}: {e}")

    # 5. Rellenar huecos y suavizar
    p(60, "Procesando datos: rellenando huecos...")
    # Rellenar NaNs con valores cercanos
    mask = np.isnan(combined_dem)
    if np.any(mask):
        combined_dem[mask] = np.nanmin(combined_dem)

    p(70, "Suavizando terreno...")
    combined_dem = gaussian_filter(combined_dem, sigma=2.0)

    # 6. Crear una malla 3D
    p(75, "Creando modelo 3D...")

    # Escalado vertical
    vertical_exaggeration = 0.1
    combined_dem = combined_dem * vertical_exaggeration

    # Crear grid de coordenadas
    y, x = np.mgrid[0:height, 0:width]
    x = x * (lon_range / width)
    y = y * (lat_range / height)

    # 7. Crear la malla con PyVista
    grid = pv.StructuredGrid(x, y, combined_dem)

    # Convertir a PolyData para poder guardar como STL
    surface = grid.extract_surface()

    # 8. Añadir base
    p(85, "Añadiendo base al modelo...")

    # Encontrar el valor mínimo de altura
    zmin = np.nanmin(combined_dem)
    base_height = zmin - 0.2  # Base ligeramente más baja

    # Extruir bordes hacia abajo para crear paredes
    edges = surface.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    walls = edges.extrude([0, 0, base_height - zmin], capping=False)

    # Crear base
    bounds = surface.bounds
    base = pv.Box([bounds[0], bounds[1], bounds[2], bounds[3], base_height, base_height])

    # Combinar todo
    model = surface + walls + base
    model = model.clean()

    # 9. Guardar el modelo STL
    p(95, "Guardando modelo...")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    p(100, f"¡Mapa completo de Ecuador generado! Guardado en: {out_path}")
    return out_path
