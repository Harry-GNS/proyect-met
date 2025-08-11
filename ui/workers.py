from PyQt6.QtCore import QThread, pyqtSignal
from core.stl_generator import generate_stl_from_hgt, generate_full_ecuador_map
from core.hgt_utils import find_hgt_files

class STLWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, str)  # success, stl_path, error_msg

    def __init__(self, hgt_path: str = None, out_dir: str = None, generate_full_map: bool = False,
                 only_ecuador: bool = True, resolution: str = "medium"):
        super().__init__()
        self.hgt_path = hgt_path
        self.out_dir = out_dir
        self.generate_full_map = generate_full_map
        self.only_ecuador = only_ecuador
        self.resolution = resolution

    def run(self):
        try:
            def cb(val, msg):
                self.progress.emit(val, msg)

            if self.generate_full_map:
                # Generar el mapa completo con las opciones seleccionadas
                stl_path = generate_full_ecuador_map(
                    progress_callback=cb,
                    resolution=self.resolution,
                    only_ecuador=self.only_ecuador
                )
            else:
                # Generar un solo tile HGT
                stl_path = generate_stl_from_hgt(self.hgt_path, self.out_dir, cb)

            self.finished.emit(True, stl_path, "")
        except Exception as e:
            self.finished.emit(False, "", str(e))