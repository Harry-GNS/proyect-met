from pathlib import Path
import os

class Paths:
    # Corregir el cálculo de la ruta raíz del proyecto
    project_root = Path(__file__).resolve().parents[1]  # Cambiado de parents[2] a parents[1]

    # Carpetas de datos de entrada
    data_dir = project_root / "data"
    hgt_dir = data_dir / "hgt"
    boundaries_dir = data_dir / "boundaries"

    # Carpetas de salida
    outputs_dir = project_root / "outputs"
    stl_dir = outputs_dir / "stl"
    previews_dir = outputs_dir / "previews"

    # Archivos específicos
    ecuador_geojson = boundaries_dir / "ecuador.geojson"

    @staticmethod
    def ensure():
        """Asegura que todas las carpetas necesarias existan"""
        for p in [
            Paths.data_dir,
            Paths.hgt_dir,
            Paths.boundaries_dir,
            Paths.outputs_dir,
            Paths.stl_dir,
            Paths.previews_dir
        ]:
            p.mkdir(parents=True, exist_ok=True)

        # Imprimir rutas para diagnóstico
        print(f"Rutas configuradas:")
        print(f"- Proyecto: {Paths.project_root}")
        print(f"- Datos HGT: {Paths.hgt_dir}")
        print(f"- Límites: {Paths.boundaries_dir}")
        print(f"- Salida STL: {Paths.stl_dir}")

    @staticmethod
    def update_stl_dir(new_path):
        """Actualiza la ruta de la carpeta de STL"""
        if not isinstance(new_path, Path):
            new_path = Path(new_path)

        Paths.stl_dir = new_path
        Paths.stl_dir.mkdir(parents=True, exist_ok=True)
        return Paths.stl_dir

    @staticmethod
    def update_hgt_dir(new_path):
        """Actualiza la ruta de la carpeta de HGT"""
        if not isinstance(new_path, Path):
            new_path = Path(new_path)

        Paths.hgt_dir = new_path
        Paths.hgt_dir.mkdir(parents=True, exist_ok=True)
        return Paths.hgt_dir
