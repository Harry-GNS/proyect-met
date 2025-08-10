import os
import numpy as np
from scipy.ndimage import gaussian_filter
import pyvista as pv
import multiprocessing as mp
from progress_bar import ProgressBarWindow

# CONFIGURACIÓN DE RENDIMIENTO - Cambia este valor para controlar el uso del CPU
CPU_USAGE_PERCENT = 75  # Porcentaje del procesador a usar (10-100)

# ------------ Funciones para paralelización ------------

def process_surface_points_chunk(args):
    """Procesar un chunk de puntos de superficie en paralelo"""
    i_start, i_end, nx, ny, X, Y, Z_smooth = args
    points_chunk = []
    for i in range(i_start, i_end):
        for j in range(ny):
            points_chunk.append([X[i, j], Y[i, j], Z_smooth[i, j]])
    return points_chunk

def process_base_points_chunk(args):
    """Procesar un chunk de puntos de base en paralelo"""
    i_start, i_end, nx, ny, X, Y, base_height = args
    points_chunk = []
    for i in range(i_start, i_end):
        for j in range(ny):
            points_chunk.append([X[i, j], Y[i, j], base_height])
    return points_chunk

def process_faces_chunk(args):
    """Procesar un chunk de caras en paralelo"""
    i_start, i_end, nx, ny, offset = args
    faces_chunk = []

    # top y base
    for i in range(i_start, min(i_end, nx - 1)):
        for j in range(ny - 1):
            p1 = i * ny + j
            p2 = i * ny + (j + 1)
            p3 = (i + 1) * ny + (j + 1)
            p4 = (i + 1) * ny + j
            faces_chunk.extend([4, p1, p2, p3, p4])

            p1b, p2b, p3b, p4b = p1 + offset, p2 + offset, p3 + offset, p4 + offset
            faces_chunk.extend([4, p4b, p3b, p2b, p1b])  # invertida

    return faces_chunk

def process_triangle_chunk(args):
    """Convertir un chunk de faces a triángulos en paralelo"""
    face_chunk_start, face_chunk_end, faces = args
    triangles_chunk = []

    for j in range(face_chunk_start, face_chunk_end, 5):  # cada cara ocupa 5 elementos [4, p1, p2, p3, p4]
        if j + 4 < len(faces):
            p1, p2, p3, p4 = faces[j+1], faces[j+2], faces[j+3], faces[j+4]
            # Dividir quad en 2 triángulos
            triangles_chunk.extend([3, p1, p2, p3])  # primer triángulo
            triangles_chunk.extend([3, p1, p3, p4])  # segundo triángulo

    return triangles_chunk

def process_hgt_file(hgt_file_path):
    """Procesar un archivo HGT en paralelo"""
    def parse_hgt_name(fname):
        base = os.path.basename(fname)
        ns = base[0].upper()
        latd = int(base[1:3])
        ew = base[3].upper()
        lond = int(base[4:7])
        lat0 = latd if ns == 'N' else -latd
        lon0 = lond if ew == 'E' else -lond
        return lat0, lon0

    def read_hgt(path):
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype='>i2')
        size = int(np.sqrt(data.size))
        if size * size != data.size:
            raise ValueError('Tamaño no cuadrado en {}'.format(path))
        Z = data.reshape((size, size)).astype(float)

        # manejar voids SRTM (-32768)
        void_mask = Z <= -32000
        if void_mask.any():
            Z[void_mask] = np.nan
            # relleno simple con el mínimo no NaN
            Z = np.where(np.isnan(Z), np.nanmin(Z), Z)
        return Z, size

    try:
        lat0, lon0 = parse_hgt_name(hgt_file_path)
        Z, size = read_hgt(hgt_file_path)
        return (lat0, lon0), (Z, size)
    except Exception as e:
        print("Error procesando {}: {}".format(hgt_file_path, e))
        return None, None

def process_grid_tile(args):
    """Procesar un tile de la rejilla en paralelo"""
    i_lat, j_lon, lat0, lon0, tiles, size = args

    if (lat0, lon0) in tiles:
        Z_tile, tile_size = tiles[(lat0, lon0)]
        block = Z_tile
        missing = False
    else:
        # tile faltante -> bloque NaN
        block = np.full((size, size), np.nan, dtype=float)
        missing = (lat0, lon0)

    # evitar duplicar bordes compartidos
    if i_lat > 0:
        block = block[1:, :]
    if j_lon > 0:
        block = block[:, 1:]

    return i_lat, j_lon, block, missing

def process_lateral_faces_chunk(args):
    """Procesar caras laterales en paralelo"""
    face_type, start_idx, end_idx, nx, ny, offset = args
    faces_chunk = []

    if face_type == 'left_right':
        # Caras izquierda y derecha
        for i in range(start_idx, end_idx):
            # Lado izquierdo (j=0)
            p1t = i * ny + 0
            p2t = (i + 1) * ny + 0
            p1b = p1t + offset
            p2b = p2t + offset
            faces_chunk.extend([4, p1t, p1b, p2b, p2t])

            # Lado derecho (j=ny-1)
            p1t = i * ny + (ny - 1)
            p2t = (i + 1) * ny + (ny - 1)
            p1b = p1t + offset
            p2b = p2t + offset
            faces_chunk.extend([4, p2t, p2b, p1b, p1t])

    elif face_type == 'top_bottom':
        # Caras superior e inferior
        for j in range(start_idx, end_idx):
            # Lado superior (i=0)
            p1t = 0 * ny + j
            p2t = 0 * ny + (j + 1)
            p1b = p1t + offset
            p2b = p2t + offset
            faces_chunk.extend([4, p1t, p2t, p2b, p1b])

            # Lado inferior (i=nx-1)
            p1t = (nx - 1) * ny + j
            p2t = (nx - 1) * ny + (j + 1)
            p1b = p1t + offset
            p2b = p2t + offset
            faces_chunk.extend([4, p2t, p1t, p1b, p2b])

    return faces_chunk

def main(hgt_file_path=None):
    # ------------ Utilidades ------------

    def parse_hgt_name(fname):
        # Ej: N00W079.hgt, S01E123.hgt
        base = os.path.basename(fname)
        ns = base[0].upper()
        latd = int(base[1:3])
        ew = base[3].upper()
        lond = int(base[4:7])
        lat0 = latd if ns == 'N' else -latd
        lon0 = lond if ew == 'E' else -lond
        return lat0, lon0

    def read_hgt(path):
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype='>i2')
        size = int(np.sqrt(data.size))
        if size * size != data.size:
            raise ValueError('Tamaño no cuadrado en {}'.format(path))
        Z = data.reshape((size, size)).astype(float)

        # manejar voids SRTM (-32768)
        void_mask = Z <= -32000
        if void_mask.any():
            Z[void_mask] = np.nan
            # relleno simple con el mínimo no NaN
            Z = np.where(np.isnan(Z), np.nanmin(Z), Z)
        return Z, size

    # ------------ 1) Recolectar y procesar HGT en paralelo ------------
    if hgt_file_path:
        hgt_files = [hgt_file_path]
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        hgt_files = []
        for r, d, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith('.hgt'):
                    hgt_files.append(os.path.join(r, f))
        if not hgt_files:
            raise FileNotFoundError('No se encontraron archivos .hgt')

    # Calcular número de procesos basado en CPU_USAGE_PERCENT
    max_processes = mp.cpu_count()
    num_processes = max(1, int(max_processes * CPU_USAGE_PERCENT / 100))
    print("Procesando {} archivos HGT con {} procesos...".format(len(hgt_files), num_processes))

    tiles = {}
    sizes = set()
    lat_set = set()
    lon_set = set()

    # Procesar archivos HGT en paralelo
    if len(hgt_files) > 1 and num_processes > 1:
        print("Usando multiprocesamiento para lectura de archivos HGT...")
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_hgt_file, hgt_files)

        for result in results:
            if result[0] is not None and result[1] is not None:
                (lat0, lon0), (Z, size) = result
                tiles[(lat0, lon0)] = (Z, size)
                sizes.add(size)
                lat_set.add(lat0)
                lon_set.add(lon0)
    else:
        # Procesamiento secuencial para pocos archivos
        print("Usando procesamiento secuencial para lectura de archivos HGT...")
        for fp in hgt_files:
            lat0, lon0 = parse_hgt_name(fp)
            Z, size = read_hgt(fp)
            tiles[(lat0, lon0)] = (Z, size)
            sizes.add(size)
            lat_set.add(lat0)
            lon_set.add(lon0)

    if len(sizes) != 1:
        raise ValueError('Hay tamaños HGT distintos: {}'.format(sizes))
    size = sizes.pop()

    # ------------ 2) Construir rejilla completa (permitiendo huecos) ------------
    lat_min, lat_max = min(lat_set), max(lat_set)
    lon_min, lon_max = min(lon_set), max(lon_set)

    # lat: norte→sur; lon: oeste→este
    lat_list = list(range(lat_max, lat_min - 1, -1))
    lon_list = list(range(lon_min, lon_max + 1))

    nlat = len(lat_list)
    nlon = len(lon_list)

    H = nlat * (size - 1) + 1
    W = nlon * (size - 1) + 1
    Z_big = np.full((H, W), np.nan, dtype=float)

    missing = []

    print("Construyendo rejilla completa...")
    for i_lat, lat0 in enumerate(lat_list):
        for j_lon, lon0 in enumerate(lon_list):
            if (lat0, lon0) in tiles:
                Z_tile, tile_size = tiles[(lat0, lon0)]
                block = Z_tile
            else:
                # tile faltante -> bloque NaN
                block = np.full((size, size), np.nan, dtype=float)
                missing.append((lat0, lon0))

            # evitar duplicar bordes compartidos
            if i_lat > 0:
                block = block[1:, :]
            if j_lon > 0:
                block = block[:, 1:]

            r0 = i_lat * (size - 1)
            c0 = j_lon * (size - 1)
            Z_big[r0:r0 + block.shape[0], c0:c0 + block.shape[1]] = block

    if missing:
        print("Tiles faltantes (rellenados): {}. Ejemplos: {}".format(len(missing), missing[:8]))

    # Rellenar NaN con el mínimo válido global (simple y robusto)
    if np.isnan(Z_big).any():
        finite_min = np.nanmin(Z_big)
        Z_big = np.where(np.isnan(Z_big), finite_min, Z_big)

    # ------------ 3) Suavizado y submuestreo optimizado ------------
    print("Aplicando suavizado gaussiano...")
    # Reducir sigma para preservar más detalles
    sigma = 5.0  # Reducido de 10.0 para preservar más detalles alrededor de cordillera
    Z = gaussian_filter(Z_big, sigma=sigma, truncate=2.5)  # truncate más pequeño para mayor detalle

    print("Aplicando submuestreo...")
    skip = 10  # Reducido de 8 a 6 para mayor resolución y más detalles
    Z = Z[::skip, ::skip]
    nx, ny = Z.shape

    print("Dimensiones finales: {}x{}".format(nx, ny))

    # Coordenadas "unitless" (ajusta escala si quieres mm/metros)
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ------------ 4) Construir sólido con base y bordes suavizados ------------
    min_height = float(np.min(Z))

    # Factor de escala vertical para reducir las alturas
    vertical_scale = 0.12  # Reducido de 0.15 a 0.12 para cordillera más baja
    Z_scaled = Z * vertical_scale

    # ESCALADO NO LINEAL PARA REDUCIR CORDILLERA Y PRESERVAR DETALLES BAJOS
    # Aplicar compresión logarítmica a las alturas más altas
    Z_max_original = np.max(Z_scaled)
    Z_min_original = np.min(Z_scaled)
    Z_range = Z_max_original - Z_min_original

    # Normalizar a rango 0-1
    Z_norm = (Z_scaled - Z_min_original) / Z_range if Z_range > 0 else Z_scaled

    # Aplicar compresión no lineal (más fuerte en alturas altas)
    compression_factor = 0.6  # Factor para comprimir montañas altas
    Z_compressed = np.where(Z_norm > 0.3,
                           0.3 + (Z_norm - 0.3) * compression_factor,  # Comprimir alturas > 30%
                           Z_norm)  # Mantener detalles bajos

    # Devolver al rango original pero con compresión
    Z_scaled = Z_min_original + Z_compressed * Z_range

    # Recalcular min_height después del escalado
    min_height = float(np.min(Z_scaled))
    base_thickness = 20.0  # Reducido para base más delgada
    base_height = min_height - base_thickness

    # NORMALIZACIÓN PARA ALTURA MÁXIMA DE 190
    max_terrain_height = 190.0  # Altura máxima deseada en el eje Z
    current_max = float(np.max(Z_scaled))
    current_min = float(np.min(Z_scaled))

    # Calcular el rango actual del terreno
    current_range = current_max - current_min

    # Escalar para que el terreno tenga exactamente 190 unidades de altura máxima
    if current_range > 0:
        # El terreno escalado irá de base_height hasta base_height + max_terrain_height
        scale_factor = max_terrain_height / current_range
        Z_normalized = base_height + (Z_scaled - current_min) * scale_factor
    else:
        Z_normalized = Z_scaled

    print("Cordillera con escalado no lineal aplicado")
    print("Altura máxima del terreno limitada a: {:.1f}".format(max_terrain_height))
    print("Rango de alturas: {:.2f} a {:.2f}".format(np.min(Z_normalized), np.max(Z_normalized)))

    edge_fade = 10
    Z_smooth = Z_normalized.copy()  # Usar Z_normalized en lugar de Z_scaled
    for i in range(edge_fade):
        Z_smooth[i, :]      = np.minimum(Z_smooth[i, :],      base_height + (Z_smooth[i, :]      - base_height) * (i / edge_fade))
        Z_smooth[-(i+1), :] = np.minimum(Z_smooth[-(i+1), :], base_height + (Z_smooth[-(i+1), :] - base_height) * (i / edge_fade))
        Z_smooth[:, i]      = np.minimum(Z_smooth[:, i],      base_height + (Z_smooth[:, i]      - base_height) * (i / edge_fade))
        Z_smooth[:, -(i+1)] = np.minimum(Z_smooth[:, -(i+1)], base_height + (Z_smooth[:, -(i+1)] - base_height) * (i / edge_fade))

    print("Factor de escala vertical aplicado: {}".format(vertical_scale))
    print("Altura mínima final: {:.2f}, Altura máxima final: {:.2f}".format(np.min(Z_smooth), np.max(Z_smooth)))

    # ------------ Procesamiento paralelo de puntos y caras ------------
    # Calcular número de procesos basado en CPU_USAGE_PERCENT
    max_processes = mp.cpu_count()
    num_processes = max(1, int(max_processes * CPU_USAGE_PERCENT / 100))  # QUITADO el límite de 4 procesos
    print("Usando {} de {} núcleos disponibles ({}% CPU)".format(num_processes, max_processes, CPU_USAGE_PERCENT))

    # Usar procesamiento secuencial si la matriz es pequeña o hay pocos procesos
    if nx * ny < 5000 or num_processes == 1:  # Reducido umbral para usar más paralelización
        print("Usando procesamiento secuencial (matriz pequeña)...")
        # Procesamiento secuencial
        points = []
        # puntos superficie
        for i in range(nx):
            for j in range(ny):
                points.append([X[i, j], Y[i, j], Z_smooth[i, j]])
        # puntos base
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
                faces.extend([4, p4b, p3b, p2b, p1b])  # invertida
    else:
        # Procesamiento paralelo
        # Dividir el trabajo en chunks
        chunk_size = max(1, nx // num_processes)
        chunks_surface = []
        chunks_base = []
        chunks_faces = []

        for i in range(0, nx, chunk_size):
            i_end = min(i + chunk_size, nx)
            chunks_surface.append((i, i_end, nx, ny, X, Y, Z_smooth))
            chunks_base.append((i, i_end, nx, ny, X, Y, base_height))
            chunks_faces.append((i, i_end, nx, ny, nx * ny))

        # Procesar puntos de superficie en paralelo
        print("Procesando puntos de superficie...")
        with mp.Pool(processes=num_processes) as pool:
            surface_results = pool.map(process_surface_points_chunk, chunks_surface)
        points = [point for chunk in surface_results for point in chunk]

        # Procesar puntos de base en paralelo
        print("Procesando puntos de base...")
        with mp.Pool(processes=num_processes) as pool:
            base_results = pool.map(process_base_points_chunk, chunks_base)
        points.extend([point for chunk in base_results for point in chunk])

        # Procesar caras en paralelo
        print("Procesando caras principales...")
        with mp.Pool(processes=num_processes) as pool:
            faces_results = pool.map(process_faces_chunk, chunks_faces)
        faces = [face for chunk in faces_results for face in chunk]

        offset = nx * ny

    # Procesar caras laterales con multiprocesamiento
    print("Procesando caras laterales con multiprocesamiento...")
    if nx > 100 and ny > 100 and num_processes > 1:  # Solo usar multiprocesamiento si vale la pena
        lateral_chunks = []

        # Dividir caras izquierda/derecha
        chunk_size_lr = max(1, (nx - 1) // num_processes)
        for i in range(0, nx - 1, chunk_size_lr):
            i_end = min(i + chunk_size_lr, nx - 1)
            lateral_chunks.append(('left_right', i, i_end, nx, ny, offset))

        # Dividir caras superior/inferior
        chunk_size_tb = max(1, (ny - 1) // num_processes)
        for j in range(0, ny - 1, chunk_size_tb):
            j_end = min(j + chunk_size_tb, ny - 1)
            lateral_chunks.append(('top_bottom', j, j_end, nx, ny, offset))

        # Procesar caras laterales en paralelo
        with mp.Pool(processes=num_processes) as pool:
            lateral_results = pool.map(process_lateral_faces_chunk, lateral_chunks)

        # Combinar resultados de caras laterales
        for chunk_result in lateral_results:
            faces.extend(chunk_result)
    else:
        # Procesamiento secuencial para datasets pequeños
        print("Usando procesamiento secuencial para caras laterales...")
        for i in range(nx - 1):
            p1t = i * ny + 0
            p2t = (i + 1) * ny + 0
            p1b = p1t + offset
            p2b = p2t + offset
            faces.extend([4, p1t, p1b, p2b, p2t])

            p1t = i * ny + (ny - 1)
            p2t = (i + 1) * ny + (ny - 1)
            p1b = p1t + offset
            p2b = p2t + offset
            faces.extend([4, p2t, p2b, p1b, p1t])

        for j in range(ny - 1):
            p1t = 0 * ny + j
            p2t = 0 * ny + (j + 1)
            p1b = p1t + offset
            p2b = p2t + offset
            faces.extend([4, p1t, p2t, p2b, p1b])

            p1t = (nx - 1) * ny + j
            p2t = (nx - 1) * ny + (j + 1)
            p1b = p1t + offset
            p2b = p2t + offset
            faces.extend([4, p2t, p1t, p1b, p2b])

    print("Creando mesh optimizado con multiprocesamiento...")

    # Convertir a arrays numpy para mayor eficiencia
    points_array = np.array(points, dtype=np.float32)  # float32 es más rápido
    print("Puntos convertidos a array: {} puntos".format(len(points_array)))

    # Convertir faces a triángulos usando multiprocesamiento
    print("Optimizando caras con multiprocesamiento...")

    # Dividir el trabajo de conversión de faces en chunks
    total_faces = len(faces)
    chunk_size = max(5000, total_faces // (num_processes * 4))  # Chunks más pequeños para mejor distribución

    if total_faces > 10000 and num_processes > 1:  # Solo usar multiprocesamiento si vale la pena
        print("Usando {} procesos para convertir {} faces a triángulos...".format(num_processes, total_faces // 5))

        triangle_chunks = []
        for i in range(0, total_faces, chunk_size):
            chunk_end = min(i + chunk_size, total_faces)
            # Asegurar que terminemos en límite de face (múltiplo de 5)
            if chunk_end < total_faces:
                chunk_end = ((chunk_end // 5) * 5)
            triangle_chunks.append((i, chunk_end, faces))

        # Procesar conversión de triángulos en paralelo
        with mp.Pool(processes=num_processes) as pool:
            triangle_results = pool.map(process_triangle_chunk, triangle_chunks)

        # Combinar resultados
        triangles = []
        for chunk_result in triangle_results:
            triangles.extend(chunk_result)
    else:
        # Procesamiento secuencial para datasets pequeños
        print("Usando procesamiento secuencial para conversión de triángulos...")
        triangles = []
        for j in range(0, total_faces, 5):  # cada cara ocupa 5 elementos [4, p1, p2, p3, p4]
            if j + 4 < total_faces:
                p1, p2, p3, p4 = faces[j+1], faces[j+2], faces[j+3], faces[j+4]
                # Dividir quad en 2 triángulos
                triangles.extend([3, p1, p2, p3])  # primer triángulo
                triangles.extend([3, p1, p3, p4])  # segundo triángulo

    print("Caras convertidas a triángulos: {} triángulos".format(len(triangles) // 4))

    # Crear mesh con configuración optimizada
    print("Construyendo mesh PyVista...")
    mesh = pv.PolyData(points_array, triangles)

    # Aplicar operaciones de limpieza más eficientes
    print("Aplicando limpieza optimizada...")
    #modificacion
    # Optimización rápida antes de crear el mesh
    print("Pre-optimizando datos...")
    points_rounded = np.round(points_array, decimals=4)
    unique_points, inverse_indices = np.unique(points_rounded, axis=0, return_inverse=True)

    if len(unique_points) < len(points_array):
        print("Eliminados {} puntos duplicados".format(len(points_array) - len(unique_points)))

        # Actualizar índices en triangles
        triangles_updated = []
        for i in range(0, len(triangles), 4):
            if i + 3 < len(triangles):
                triangles_updated.extend([3,
                                        inverse_indices[triangles[i+1]],
                                        inverse_indices[triangles[i+2]],
                                        inverse_indices[triangles[i+3]]])
        triangles = triangles_updated
        points_array = unique_points

    # Crear mesh con datos ya optimizados
    mesh = pv.PolyData(points_array, triangles)

    # Limpieza ligera y rápida
    print("Aplicando limpieza ligera...")
    mesh = mesh.clean(tolerance=1e-3)
    # fin de la modificacion
    #mesh = mesh.clean(tolerance=1e-6)

    print("Mesh creado exitosamente: {} puntos, {} celdas".format(mesh.n_points, mesh.n_cells))

    # ------------ 5) Guardar STL ------------
    out_stl = 'ecuador_terreno_solido.stl'
    mesh.save(out_stl)
    print("STL guardado: {}".format(out_stl))

    # ------------ 6) (Opcional) Visualizar ------------
    print("Mostrando visualización...")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, cmap='terrain', show_edges=False, lighting=True)
    plotter.enable_eye_dome_lighting()
    plotter.show_grid()
    plotter.set_scale(zscale=1.0)
    plotter.show(title='Terreno 3D Sólido (Mosaico)')

if __name__ == '__main__':
    mp.freeze_support()
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
