import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
from ecuador_boundary import get_ecuador_geojson
from preview_generator import generate_hgt_preview, generate_all_previews
from preview_3d import generate_3d_preview

# Extrae latitud y longitud del nombre de archivo HGT
HGT_PATTERN = re.compile(r'([NS])(\d{2})([EW])(\d{3})\.hgt$')

def parse_hgt_filename(filename):
    match = HGT_PATTERN.search(filename)
    if not match:
        return None
    lat_sign = 1 if match.group(1) == 'N' else -1
    lat = lat_sign * int(match.group(2))
    lon_sign = 1 if match.group(3) == 'E' else -1
    lon = lon_sign * int(match.group(4))
    return lat, lon

def find_hgt_files(root_dir):
    hgt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.hgt'):
                coords = parse_hgt_filename(fname)
                if coords:
                    hgt_files.append({
                        'path': os.path.join(dirpath, fname),
                        'lat': coords[0],
                        'lon': coords[1],
                        'name': fname
                    })
    return hgt_files

def select_hgt_by_map(root_dir='Datos'):
    hgt_files = find_hgt_files(root_dir)
    if not hgt_files:
        print('No se encontraron archivos HGT.')
        return None
    geojson_path = get_ecuador_geojson()
    if geojson_path:
        gdf = gpd.read_file(geojson_path)
        # Filtrar archivos HGT que estén dentro de Ecuador
        def hgt_in_ecuador(lat, lon, gdf):
            # El archivo cubre de (lat, lon) a (lat+1, lon+1)
            from shapely.geometry import box
            tile = box(lon, lat, lon+1, lat+1)
            return gdf.intersects(tile).any()
        hgt_files = [f for f in hgt_files if hgt_in_ecuador(f['lat'], f['lon'], gdf)]
    if not hgt_files:
        print('No hay archivos HGT dentro de Ecuador.')
        return None
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_title('Selecciona una zona HGT (clic en el cuadro)')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_aspect('equal')
    # Mostrar contorno de Ecuador
    if geojson_path:
        gdf.boundary.plot(ax=ax, color='green', linewidth=2)
    # Determinar límites
    lats = [f['lat'] for f in hgt_files]
    lons = [f['lon'] for f in hgt_files]
    ax.set_xlim(min(lons)-1, max(lons)+1)
    ax.set_ylim(min(lats)-1, max(lats)+1)
    rectangles = []
    for f in hgt_files:
        rect = Rectangle((f['lon'], f['lat']), 1, 1, edgecolor='black', facecolor='lightgray', alpha=0.7)
        ax.add_patch(rect)
        rectangles.append((rect, f))
        ax.text(f['lon']+0.5, f['lat']+0.5, f['name'], ha='center', va='center', fontsize=7)
    selected = {'file': None}
    def on_click(event):
        for rect, info in rectangles:
            if rect.contains_point((event.x, event.y)):
                selected['file'] = info['path']
                # Mostrar modelo 3D interactivo en PyVista y mapa de calor como overlay en una esquina
                import subprocess
                import sys
                # Ejecutar Mapa3DPrint.py como script, pasando el archivo seleccionado
                subprocess.run([sys.executable, 'Mapa3DPrint.py', info['path']])
                plt.close(fig)
                break
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return selected['file']
