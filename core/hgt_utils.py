import os
import re
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from core.paths import Paths
from core.ecuador_boundary import get_ecuador_geojson

HGT_PATTERN = re.compile(r'([NS])(\d{2})([EW])(\d{3})\.hgt$', re.IGNORECASE)

def parse_hgt_filename(fname: str):
    m = HGT_PATTERN.search(os.path.basename(fname))
    if not m:
        return None
    lat = int(m.group(2)) * (1 if m.group(1).upper() == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3).upper() == "E" else -1)
    return lat, lon

def find_hgt_files(root_dir=None, only_ecuador=True):
    """
    Busca archivos HGT en el directorio especificado.

    Args:
        root_dir: Directorio donde buscar archivos HGT. Si es None, usa Paths.hgt_dir
        only_ecuador: Si True, filtra solo archivos que intersectan con Ecuador

    Returns:
        Lista de diccionarios con información de los archivos HGT encontrados
    """
    if root_dir is None:
        root_dir = str(Paths.hgt_dir)

    print(f"Buscando archivos HGT en: {root_dir}")

    hgt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".hgt"):
                coords = parse_hgt_filename(fn)
                if coords:
                    hgt_files.append({
                        "path": os.path.join(dirpath, fn),
                        "lat": coords[0],
                        "lon": coords[1],
                        "name": fn
                    })

    print(f"Total de archivos HGT encontrados: {len(hgt_files)}")

    # Filtrar por Ecuador solo si se especifica y hay archivos
    if only_ecuador and hgt_files:
        try:
            geojson = get_ecuador_geojson()
            if not geojson:
                print("No se pudo obtener el GeoJSON de Ecuador. Mostrando todos los archivos HGT.")
                return hgt_files

            print(f"Filtrando archivos HGT usando: {geojson}")
            gdf = gpd.read_file(geojson)

            def in_ec(lat, lon):
                tile = box(lon, lat, lon+1, lat+1)
                return gdf.intersects(tile).any()

            ecuador_files = [f for f in hgt_files if in_ec(f["lat"], f["lon"])]
            print(f"Archivos HGT en Ecuador: {len(ecuador_files)}")
            return ecuador_files

        except Exception as e:
            print(f"Error al filtrar archivos por Ecuador: {e}")
            # En caso de error, retorna todos los archivos
            return hgt_files

    return hgt_files

def read_hgt_array(hgt_path: str):
    with open(hgt_path, "rb") as f:
        data = np.fromfile(f, dtype=">i2")
    size = int(np.sqrt(data.size))
    if size * size != data.size:
        raise ValueError(f"Archivo HGT no cuadrado: {hgt_path}")
    Z = data.reshape((size, size)).astype(float)
    # manejar voids SRTM
    void_mask = Z <= -32000
    if void_mask.any():
        Z[void_mask] = np.nan
        Z = np.where(np.isnan(Z), np.nanmin(Z), Z)
    return Z