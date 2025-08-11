import os
import requests
from pathlib import Path
import json
import time
from core.paths import Paths

# Múltiples URLs para mayor confiabilidad
ECU_URLS = [
    "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries/ECU.geojson",
    "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/ECU.geo.json"
]

# GeoJSON básico de Ecuador como respaldo (simplificado)
BASIC_ECUADOR_GEOJSON = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"name": "Ecuador"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-81.0, -5.0], [-81.0, 1.5],
                [-75.0, 1.5], [-75.0, -5.0],
                [-81.0, -5.0]
            ]]
        }
    }]
}

def get_ecuador_geojson():
    """
    Obtiene el archivo GeoJSON de Ecuador.
    Primero intenta usar un archivo local, luego descarga desde internet,
    y finalmente usa un contorno básico como respaldo si todo lo demás falla.
    """
    Paths.ensure()
    out_path = Paths.ecuador_geojson

    # 1. Verificar si el archivo ya existe localmente
    if out_path.exists():
        print(f"Usando archivo GeoJSON existente: {out_path}")
        return str(out_path)

    print("El archivo GeoJSON de Ecuador no existe localmente, intentando descargar...")

    # 2. Intentar descargar de múltiples URLs
    for i, url in enumerate(ECU_URLS):
        try:
            print(f"Intentando descargar desde {url}...")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                print(f"GeoJSON de Ecuador guardado en {out_path}")
                return str(out_path)
            else:
                print(f"Error al descargar (HTTP {r.status_code})")
        except Exception as e:
            print(f"Error al descargar: {e}")
            if i < len(ECU_URLS) - 1:  # Si no es el último intento
                print("Intentando con URL alternativa...")
                time.sleep(1)  # Pequeña pausa antes del siguiente intento

    # 3. Usar GeoJSON básico como último recurso
    print("Usando contorno básico de Ecuador como respaldo")
    with open(out_path, 'w') as f:
        json.dump(BASIC_ECUADOR_GEOJSON, f)

    return str(out_path)