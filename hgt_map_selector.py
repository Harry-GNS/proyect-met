import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
from ecuador_boundary import get_ecuador_geojson
from preview_generator import generate_hgt_preview, generate_all_previews

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
    # Generar previews si no existen
    generate_all_previews(root_dir, 'previews')
    geojson_path = get_ecuador_geojson()
    if geojson_path:
        gdf = gpd.read_file(geojson_path)
    # Mostrar preview general de alturas
    preview_paths = [generate_hgt_preview(f['path'], 'previews') for f in hgt_files]
    fig, (ax, ax_preview) = plt.subplots(1, 2, figsize=(14, 8))
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
    # Mostrar preview general (mosaico de previews)
    ax_preview.set_title('Mapa de calor de alturas (preview)')
    ax_preview.axis('off')
    # Mostrar previews individuales en forma de mosaico
    n = len(preview_paths)
    for i, img_path in enumerate(preview_paths):
        img = plt.imread(img_path)
        x = i % 5
        y = i // 5
        ax_preview.imshow(img, extent=[x, x+1, y, y+1], aspect='auto')
    selected = {'file': None}
    def on_click(event):
        for rect, info in rectangles:
            if rect.contains_point((event.x, event.y)):
                selected['file'] = info['path']
                # Mostrar preview individual al seleccionar
                preview_path = generate_hgt_preview(info['path'], 'previews')
                plt.figure(figsize=(4, 4))
                plt.title(f'Preview de alturas: {info["name"]}')
                img = plt.imread(preview_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                plt.close(fig)
                break
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return selected['file']
