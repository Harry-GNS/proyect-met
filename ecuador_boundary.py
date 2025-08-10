import os
import requests

def get_ecuador_geojson(geojson_path='ecuador.geojson'):
    if not os.path.exists(geojson_path):
        url = 'https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries/ECU.geojson'
        print('Descargando el contorno de Ecuador...')
        r = requests.get(url)
        if r.status_code == 200:
            with open(geojson_path, 'wb') as f:
                f.write(r.content)
            print('GeoJSON de Ecuador guardado como', geojson_path)
        else:
            print('No se pudo descargar el GeoJSON de Ecuador.')
            return None
    return geojson_path
