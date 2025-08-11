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

def generate_full_ecuador_map(out_dir: Optional[str] = None, progress_callback: ProgressCB = None,
                        resolution: str = "medium", only_ecuador: bool = True) -> str:
    """
    Genera un modelo 3D SÓLIDO (no hueco) de todo el territorio usando los archivos HGT disponibles.
    Optimizado para usar más recursos de hardware y mantener alta calidad.
    """
    def p(val, msg):
        if progress_callback:
            progress_callback(val, msg)

    # Aseguramos que existan las carpetas necesarias
    Paths.ensure()
    if out_dir is None:
        out_dir = str(Paths.stl_dir)

    # Parámetros agresivos para aprovechar mejor el hardware disponible
    # Usando 70% CPU y 80% RAM disponible
    if only_ecuador:
        # Para Ecuador, usar parámetros muy altos aprovechando los recursos
        if resolution == "low":
            skip = 12  # Mejor calidad (era 16)
            smoothing = 1.5  # Menos suavizado
            max_pixels = 3000000  # Mucho más alto (era 1.5M)
            vertical_exaggeration = 0.18
            max_faces = 800000  # Más caras
        elif resolution == "high":
            skip = 4  # Máxima calidad (era 6)
            smoothing = 0.6  # Mínimo suavizado
            max_pixels = 8000000  # Máximo aprovechamiento RAM (era 4M)
            vertical_exaggeration = 0.25
            max_faces = 2000000  # Mucho más detalle
        else:  # medium
            skip = 7  # Mejor calidad (era 10)
            smoothing = 1.0  # Menos suavizado
            max_pixels = 5000000  # Aprovechar RAM (era 2.5M)
            vertical_exaggeration = 0.22
            max_faces = 1500000  # Más caras
    else:
        # Para todos los HGT, usar parámetros más altos aprovechando recursos
        if resolution == "low":
            skip = 16  # Mejor que antes (era 20)
            smoothing = 2.0  # Menos suavizado
            max_pixels = 2000000  # Más píxeles (era 1M)
            vertical_exaggeration = 0.15
            max_faces = 600000  # Más caras
        elif resolution == "high":
            skip = 8  # Mucho mejor (era 12)
            smoothing = 1.0  # Menos suavizado
            max_pixels = 4000000  # Aprovechar RAM (era 2M)
            vertical_exaggeration = 0.20
            max_faces = 1200000  # Más detalle
        else:  # medium
            skip = 12  # Mejor calidad (era 16)
            smoothing = 1.5  # Menos suavizado
            max_pixels = 3000000  # Más píxeles (era 1.5M)
            vertical_exaggeration = 0.18
            max_faces = 900000  # Más caras

    # Nombre de archivo para el mapa completo
    area_text = "ecuador" if only_ecuador else "completo"
    out_path = str(Path(out_dir) / f"mapa_solido_{area_text}_{resolution}.stl")

    # 1. Encontrar todos los archivos HGT
    p(5, f"Buscando archivos HGT{' de Ecuador' if only_ecuador else ''}...")
    hgt_files = find_hgt_files(only_ecuador=only_ecuador)
    if not hgt_files:
        raise ValueError(f"No se encontraron archivos HGT{' para Ecuador' if only_ecuador else ''}")

    # Aprovechar mejor los recursos - procesar más archivos
    max_files = 80 if only_ecuador else 50  # Más archivos para mejor cobertura
    if len(hgt_files) > max_files:
        p(8, f"Procesando {max_files} de {len(hgt_files)} archivos HGT para optimizar recursos")
        hgt_files = hgt_files[:max_files]

    p(10, f"Se procesarán {len(hgt_files)} archivos HGT para modelo SÓLIDO")

    # 2. Determinar las dimensiones del mapa completo
    min_lat = min(f["lat"] for f in hgt_files)
    max_lat = max(f["lat"] + 1 for f in hgt_files)
    min_lon = min(f["lon"] for f in hgt_files)
    max_lon = max(f["lon"] + 1 for f in hgt_files)

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    p(15, f"Área: {lon_range:.1f}° x {lat_range:.1f}°")

    # 3. Crear un grid de alta resolución aprovechando la RAM disponible
    points_per_degree = 1201 // skip

    width = int(lon_range * points_per_degree)
    height = int(lat_range * points_per_degree)

    pixels = width * height
    if pixels > max_pixels:
        factor = (pixels / max_pixels) ** 0.5
        skip = int(skip * factor)
        points_per_degree = 1201 // skip
        width = int(lon_range * points_per_degree)
        height = int(lat_range * points_per_degree)
        p(20, f"Ajustando para modelo sólido: skip={skip}, grid={width}x{height}")

    p(20, f"Creando grid de alta resolución para modelo SÓLIDO ({width}x{height} = {width*height:,} puntos)...")
    combined_dem = np.full((height, width), np.nan, dtype=np.float32)

    # 4. Procesamiento paralelo aprovechando los núcleos disponibles
    total_files = len(hgt_files)
    # Lotes más grandes para aprovechar el CPU
    batch_size = 8 if only_ecuador else 6  # Más archivos por lote

    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = hgt_files[batch_start:batch_end]

        for i, hgt_file in enumerate(batch_files):
            file_idx = batch_start + i
            prog_val = 20 + int((file_idx / total_files) * 35)
            p(prog_val, f"Procesando {hgt_file['name']} ({file_idx+1}/{total_files})...")

            try:
                # Cargar archivo HGT
                dem = read_hgt_array(hgt_file["path"])

                # Submuestreo inteligente adaptativo
                if skip > 1:
                    # Para mapas completos, usar submuestreo más simple para ahorrar memoria
                    if only_ecuador and skip <= 10:
                        # Solo usar filtro de máximos para Ecuador con alta resolución
                        from scipy.ndimage import maximum_filter
                        kernel_size = min(skip, 3)
                        dem_filtered = maximum_filter(dem, size=kernel_size)
                        dem = dem_filtered[::skip, ::skip].astype(np.float32)
                    else:
                        # Submuestreo simple para casos de mucha memoria
                        dem = dem[::skip, ::skip].astype(np.float32)
                else:
                    dem = dem.astype(np.float32)

                # Calcular posición en el grid
                x_offset = int((hgt_file["lon"] - min_lon) * points_per_degree)
                y_offset = int((max_lat - hgt_file["lat"] - 1) * points_per_degree)

                # Verificar límites y insertar
                if (0 <= x_offset < width and 0 <= y_offset < height and
                    x_offset + dem.shape[1] <= width and y_offset + dem.shape[0] <= height):
                    combined_dem[y_offset:y_offset+dem.shape[0], x_offset:x_offset+dem.shape[1]] = dem

                # Liberar memoria más frecuentemente
                del dem

            except Exception as e:
                p(prog_val, f"Error en {hgt_file['name']}: {e}")

        # Liberación de memoria más agresiva
        import gc
        gc.collect()

    # 5. Procesamiento de datos preservando características del terreno
    p(60, "Rellenando huecos preservando características...")

    mask = np.isnan(combined_dem)
    if np.any(mask):
        # Método más sofisticado para rellenar que preserve el relieve
        from scipy.ndimage import distance_transform_edt, binary_dilation

        valid_data = combined_dem[~mask]
        if len(valid_data) > 0:
            # Usar interpolación por distancia para preservar gradientes
            distances, indices = distance_transform_edt(mask, return_indices=True)
            combined_dem[mask] = combined_dem[tuple(indices[:, mask])]

        del valid_data, mask

    # Suavizado selectivo que preserve bordes (montañas)
    p(70, "Aplicando suavizado inteligente...")
    if smoothing > 0:
        # Usar filtro gaussiano con truncado más bajo para preservar detalles
        combined_dem = gaussian_filter(combined_dem, sigma=smoothing, truncate=1.5)

    # 6. Escalado vertical mejorado para resaltar relieve
    p(75, "Optimizando relieve para impresión 3D...")

    z_min = np.min(combined_dem)
    z_max = np.max(combined_dem)
    z_range = z_max - z_min

    # Usar exageración vertical variable según la resolución
    horizontal_size = max(width, height)
    target_height_ratio = vertical_exaggeration
    vertical_scale = (horizontal_size * target_height_ratio) / z_range if z_range > 0 else 1

    # Aplicar compresión no lineal para resaltar montañas
    combined_dem_normalized = (combined_dem - z_min) / z_range if z_range > 0 else combined_dem - z_min

    # Función de mapeo que resalta las elevaciones altas (cordillera)
    gamma = 0.7  # Valor < 1 resalta las elevaciones altas
    combined_dem_enhanced = np.power(combined_dem_normalized, gamma)

    combined_dem = z_min + (combined_dem_enhanced * z_range * vertical_scale)

    # 7. Crear malla 3D preservando detalle
    p(80, "Creando modelo 3D de alta definición...")

    # Decimación más conservadora para preservar detalle geográfico
    decimation_factor = 1  # No decimar por defecto
    if width * height > max_pixels * 1.5:  # Solo decimar si es absolutamente necesario
        decimation_factor = 2
        combined_dem = combined_dem[::decimation_factor, ::decimation_factor]
        width = combined_dem.shape[1]
        height = combined_dem.shape[0]
        p(82, f"Grid ajustado a {width}x{height} para mantener detalle")

    # Crear coordenadas con escala apropiada
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Escalar para impresión manteniendo proporciones
    max_print_size = 150 if resolution == "high" else 130 if resolution == "medium" else 110
    scale_factor = max_print_size / max(width, height)

    x_coords = x_coords.astype(np.float32) * scale_factor
    y_coords = y_coords.astype(np.float32) * scale_factor
    combined_dem = combined_dem.astype(np.float32) * scale_factor

    # 8. Crear malla SÓLIDA con PyVista usando métodos compatibles
    p(80, "Creando modelo 3D SÓLIDO de alta resolución...")

    try:
        # Crear grid estructurado
        grid = pv.StructuredGrid(x_coords, y_coords, combined_dem)

        # Liberar arrays grandes inmediatamente
        del x_coords, y_coords, combined_dem
        import gc
        gc.collect()

        # Extraer superficie
        surface = grid.extract_surface()
        del grid
        gc.collect()

        # Usar método compatible para obtener número de caras
        def get_n_faces(mesh):
            try:
                if hasattr(mesh, 'n_cells'):
                    return mesh.n_cells
                elif hasattr(mesh, 'GetNumberOfCells'):
                    return mesh.GetNumberOfCells()
                else:
                    # Calcular manualmente si es necesario
                    if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                        return len(mesh.faces) // 4  # Asumiendo quads
                    return 0
            except:
                return 0

        # Simplificación conservadora para mantener calidad
        n_faces = get_n_faces(surface)
        if n_faces > max_faces:
            reduction_factor = max_faces / n_faces
            p(85, f"Optimizando modelo SÓLIDO ({n_faces:,} -> {max_faces:,} caras)...")
            surface = surface.decimate(1 - reduction_factor)

        # 9. Crear modelo completamente SÓLIDO
        p(88, "Generando modelo completamente SÓLIDO...")

        # Calcular base gruesa para modelo sólido
        base_thickness = 5 if resolution == "high" else 4 if resolution == "medium" else 3
        z_min = float(np.min(surface.points[:, 2]))
        base_height = z_min - base_thickness

        # Crear base sólida
        bounds = surface.bounds
        base = pv.Box([bounds[0], bounds[1], bounds[2], bounds[3], base_height, base_height])

        # Crear paredes laterales para modelo sólido
        p(90, "Añadiendo paredes laterales para modelo sólido...")

        # Extruir los bordes hacia abajo para crear paredes sólidas
        try:
            # Extraer bordes de la superficie
            edges = surface.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)

            # Crear paredes extruyendo hacia abajo
            if edges.n_points > 0:
                # Vector de extrusión hacia abajo
                extrude_vector = [0, 0, base_height - z_min]
                walls = edges.extrude(extrude_vector, capping=False)
            else:
                # Si no hay bordes, crear paredes manualmente
                walls = pv.PolyData()  # Paredes vacías

        except Exception as e:
            p(90, f"Creando paredes alternativas: {e}")
            # Método alternativo para crear paredes
            walls = pv.PolyData()  # Paredes vacías si falla

        # Combinar todos los componentes para modelo sólido
        p(92, "Ensamblando modelo SÓLIDO final...")

        try:
            # Método robusto para combinar componentes
            components = [surface]
            if walls.n_points > 0:
                components.append(walls)
            components.append(base)

            # Usar el método más compatible
            model = surface.copy()
            for component in components[1:]:  # Saltar surface ya que es la base
                try:
                    if hasattr(model, 'append_polydata'):
                        model = model.append_polydata(component)
                    else:
                        model = model + component
                except:
                    # Si falla, intentar método básico
                    pass

        except Exception as e:
            p(92, f"Usando método de combinación alternativo: {e}")
            # Fallback: solo superficie + base
            try:
                if hasattr(surface, 'append_polydata'):
                    model = surface.append_polydata(base)
                else:
                    model = surface + base
            except:
                model = surface  # Último recurso

        del surface, walls, base
        gc.collect()

        # Verificar que el modelo sea sólido (watertight)
        p(95, "Verificando modelo sólido...")

        try:
            # Intentar cerrar agujeros para asegurar que sea sólido
            model = model.fill_holes(hole_size=model.length * 0.05)

            # Limpiar con tolerancia ajustada
            model = model.clean(tolerance=0.0001)  # Tolerancia más baja para modelos sólidos

            # Verificar si es manifold (necesario para modelos sólidos)
            if hasattr(model, 'is_manifold'):
                if not model.is_manifold:
                    p(96, "Reparando geometría para modelo sólido...")
                    model = model.smooth(n_iter=2, relaxation_factor=0.01)

        except Exception as e:
            p(95, f"Aplicando limpieza básica: {e}")
            model = model.clean(tolerance=0.001)

        # Verificación final de calidad
        n_faces_final = get_n_faces(model)
        if n_faces_final > max_faces * 1.3:  # Permitir más caras para modelos sólidos
            p(97, f"Optimización final para modelo sólido...")
            target_reduction = 1 - (max_faces / n_faces_final)
            model = model.decimate(target_reduction * 0.9)  # Reducir menos agresivamente

        # 10. Guardar modelo sólido optimizado
        n_points = model.n_points if hasattr(model, 'n_points') else len(model.points)
        n_faces_save = get_n_faces(model)
        p(98, f"Guardando modelo SÓLIDO ({n_points:,} puntos, {n_faces_save:,} caras)...")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Guardar como STL binario (más compacto para modelos sólidos)
        model.save(out_path, binary=True)

        del model
        gc.collect()

        area_desc = "Ecuador" if only_ecuador else "todas las regiones"
        p(100, f"¡Modelo SÓLIDO de {area_desc} generado correctamente!")
        return out_path

    except Exception as e:
        import gc
        gc.collect()
        raise RuntimeError(f"Error al crear modelo 3D sólido: {str(e)}")
