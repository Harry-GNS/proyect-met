from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                         QLabel, QFileDialog, QSizePolicy, QFrame, QComboBox)
from PyQt6.QtCore import Qt
from pyvistaqt import QtInteractor
import pyvista as pv
from pathlib import Path
import os
from core.paths import Paths

class Viewer3D(QWidget):
    def __init__(self, back_cb):
        super().__init__()
        self.back_cb = back_cb
        self.current_file = None

        # Crear layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Panel superior con controles
        top_panel = QFrame()
        top_panel.setFrameShape(QFrame.Shape.StyledPanel)
        top_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        top_layout = QHBoxLayout(top_panel)

        # Botón de volver al menú
        self.back_btn = QPushButton("← Volver al Menú")
        self.back_btn.setMinimumWidth(150)
        self.back_btn.clicked.connect(self.back_cb)

        # Información del archivo
        self.info = QLabel("")
        self.info.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Controles adicionales
        self.save_btn = QPushButton("Guardar STL como...")
        self.save_btn.setMinimumWidth(150)
        self.save_btn.clicked.connect(self._save_stl)

        # Selector de estilo de visualización
        self.view_style = QComboBox()
        self.view_style.addItems(["Terreno", "Sólido", "Wireframe", "Puntos"])
        self.view_style.currentIndexChanged.connect(self._change_view_style)

        # Agregar widgets al panel superior
        top_layout.addWidget(self.back_btn)
        top_layout.addWidget(self.info, 1)  # 1 = stretch factor
        top_layout.addWidget(self.view_style)
        top_layout.addWidget(self.save_btn)

        # Panel de visualización 3D
        viewer_panel = QFrame()
        viewer_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        # Inicializar plotter de PyVista
        self.plotter = QtInteractor(self)
        viewer_layout.addWidget(self.plotter.interactor)

        # Panel inferior con información adicional
        bottom_panel = QFrame()
        bottom_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bottom_layout = QHBoxLayout(bottom_panel)

        self.model_info = QLabel("")
        self.model_info.setStyleSheet("font-style: italic;")
        bottom_layout.addWidget(self.model_info)

        # Agregar paneles al layout principal
        main_layout.addWidget(top_panel)
        main_layout.addWidget(viewer_panel, 1)  # 1 = stretch factor
        main_layout.addWidget(bottom_panel)

    def load_stl(self, stl_path: str):
        """Carga un archivo STL y lo muestra en el visualizador 3D"""
        self.current_file = stl_path
        self.plotter.clear()

        try:
            # Cargar mesh
            mesh = pv.read(stl_path)

            # Mostrar el modelo con el estilo por defecto (Terreno)
            self._display_mesh(mesh, style="terrain")

            # Actualizar información
            file_name = Path(stl_path).name
            self.info.setText(f"Modelo: {file_name}")

            # Mostrar estadísticas del modelo
            n_points = mesh.n_points
            n_cells = mesh.n_cells
            self.model_info.setText(f"Puntos: {n_points:,} | Caras: {n_cells:,} | Archivo: {os.path.abspath(stl_path)}")

        except Exception as e:
            self.info.setText(f"Error al cargar el modelo: {str(e)}")
            self.model_info.setText("")

    def _change_view_style(self, index):
        """Cambia el estilo de visualización del modelo"""
        if self.current_file is None:
            return

        # Volver a cargar el mesh
        mesh = pv.read(self.current_file)

        # Aplicar el estilo seleccionado
        styles = ["terrain", "default", "wireframe", "points"]
        selected_style = styles[index] if index < len(styles) else "terrain"

        self._display_mesh(mesh, style=selected_style)

    def _display_mesh(self, mesh, style="terrain"):
        """Muestra un mesh con el estilo especificado"""
        self.plotter.clear()

        if style == "terrain":
            # Estilo de terreno con colores según elevación
            self.plotter.add_mesh(mesh, cmap="terrain", show_edges=False, lighting=True)
            self.plotter.enable_eye_dome_lighting()
        elif style == "wireframe":
            # Estilo de wireframe
            self.plotter.add_mesh(mesh, style="wireframe", color="black", line_width=1)
        elif style == "points":
            # Estilo de puntos
            self.plotter.add_mesh(mesh, style="points", color="blue", point_size=5)
        else:
            # Estilo por defecto (sólido)
            self.plotter.add_mesh(mesh, color="lightgray", show_edges=True)

        self.plotter.show_grid()
        self.plotter.reset_camera()

    def _save_stl(self):
        """Guarda el modelo STL actual en una nueva ubicación"""
        if self.current_file is None:
            return

        original_filename = Path(self.current_file).stem

        # Mostrar diálogo para guardar archivo
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar STL como",
            str(Paths.stl_dir / f"{original_filename}_copy.stl"),
            "STL Files (*.stl)"
        )

        if file_path:
            try:
                # Leer el mesh original y guardarlo en la nueva ubicación
                mesh = pv.read(self.current_file)
                mesh.save(file_path)

                # Actualizar información
                self.model_info.setText(f"Guardado como: {file_path}")
            except Exception as e:
                self.model_info.setText(f"Error al guardar: {str(e)}")
