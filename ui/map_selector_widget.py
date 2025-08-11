from typing import Callable, List, Tuple
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import geopandas as gpd
from core.hgt_utils import find_hgt_files
from core.ecuador_boundary import get_ecuador_geojson
from core.paths import Paths

class MapSelectorWidget(QWidget):
    def __init__(self, on_selected: Callable[[str], None], on_back: Callable[[], None], hgt_root=None):
        super().__init__()
        self.on_selected = on_selected
        self.hgt_root = hgt_root
        top = QHBoxLayout()
        self.back_btn = QPushButton("Volver")
        top.addWidget(self.back_btn)
        self.title = QLabel("Selecciona una zona HGT (clic)")
        top.addWidget(self.title)
        top.addStretch()
        layout = QVBoxLayout(self)
        layout.addLayout(top)

        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Ecuador y tiles HGT")
        self.ax.set_xlabel("Longitud")
        self.ax.set_ylabel("Latitud")
        self.ax.set_aspect("equal")

        self.rectangles: List[Tuple[Rectangle, dict]] = []
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.back_btn.clicked.connect(on_back)

    def load(self):
        self.ax.clear()
        self.ax.set_aspect("equal")

        # Mostrar ruta actual de búsqueda HGT
        print(f"Buscando archivos HGT en: {self.hgt_root or Paths.hgt_dir}")

        # Ecuador boundary
        gp = get_ecuador_geojson()
        if gp:
            print(f"Archivo GeoJSON cargado: {gp}")
            try:
                gdf = gpd.read_file(gp)
                gdf.boundary.plot(ax=self.ax, color="green", linewidth=2)
            except Exception as e:
                print(f"Error al cargar GeoJSON: {e}")
                self.ax.set_title("Error al cargar límites de Ecuador")
                self.canvas.draw()
                return

        # HGT tiles
        try:
            files = find_hgt_files(self.hgt_root, only_ecuador=True)
            print(f"Archivos HGT encontrados: {len(files)}")

            if not files:
                print("No se encontraron archivos HGT en Ecuador")
                self.ax.set_title("No se encontraron HGT en Ecuador")
                self.canvas.draw()
                return

        except Exception as e:
            print(f"Error al buscar archivos HGT: {e}")
            self.ax.set_title(f"Error al buscar archivos HGT: {str(e)}")
            self.canvas.draw()
            return

        lats = [f["lat"] for f in files]
        lons = [f["lon"] for f in files]
        self.ax.set_xlim(min(lons)-1, max(lons)+1)
        self.ax.set_ylim(min(lats)-1, max(lats)+1)

        self.rectangles.clear()
        for f in files:
            rect = Rectangle((f['lon'], f['lat']), 1, 1, edgecolor='black',
                             facecolor='lightgray', alpha=0.7, picker=True)
            self.ax.add_patch(rect)
            self.ax.text(f['lon']+0.5, f['lat']+0.5, f['name'], ha='center', va='center', fontsize=7)
            self.rectangles.append((rect, f))

        self.canvas.draw()

    def _on_pick(self, event):
        artist = event.artist
        for rect, info in self.rectangles:
            if artist == rect:
                self.on_selected(info["path"])
                break