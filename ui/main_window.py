from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog,
                         QStackedWidget, QMessageBox, QLabel, QHBoxLayout, QDialog,
                         QRadioButton, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from ui.map_selector_widget import MapSelectorWidget
from ui.progress_dialog import ProgressDialog
from ui.viewer_3d import Viewer3D
from ui.workers import STLWorker
from core.paths import Paths
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mapa 3D Ecuador - Generador de STL")
        self.resize(1200, 800)
        Paths.ensure()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Main menu
        self.menu_page = QWidget()
        self.create_main_menu()
        self.stack.addWidget(self.menu_page)

        # Map page
        self.map_page = MapSelectorWidget(on_selected=self._on_tile_selected, on_back=self._go_menu)
        self.stack.addWidget(self.map_page)

        # Viewer page
        self.viewer_page = Viewer3D(back_cb=self._go_menu)
        self.stack.addWidget(self.viewer_page)

        # Variables de estado
        self.progress_dlg = None
        self.worker = None

        # Mostrar menú principal al inicio
        self._go_menu()

    def create_main_menu(self):
        layout = QVBoxLayout(self.menu_page)

        # Título
        title_label = QLabel("Generador 3D de Terreno - Ecuador")
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #2c3e50; margin: 20px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Descripción
        desc_label = QLabel("Seleccione una opción para comenzar:")
        desc_label.setStyleSheet("font-size: 14pt; margin-bottom: 20px;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        # Contenedor principal para los botones
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)

        # Estilo común para botones
        button_style = """
        QPushButton {
            font-size: 14pt;
            padding: 12px;
            margin: 10px 50px;
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            min-width: 300px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        """

        # Botones principales
        self.btn_map = QPushButton("Seleccionar Zona en Mapa")
        self.btn_map.setStyleSheet(button_style)

        # Nuevo botón para generar todo el mapa
        self.btn_full_map = QPushButton("Generar Todo el Mapa de Ecuador")
        self.btn_full_map.setStyleSheet(button_style.replace("#3498db", "#27ae60").replace("#2980b9", "#219955"))

        self.btn_open_stl = QPushButton("Abrir Modelo STL Existente")
        self.btn_open_stl.setStyleSheet(button_style)
        self.btn_settings = QPushButton("Configurar Carpetas de Datos")
        self.btn_settings.setStyleSheet(button_style)
        self.btn_exit = QPushButton("Salir")
        self.btn_exit.setStyleSheet(button_style.replace("#3498db", "#e74c3c").replace("#2980b9", "#c0392b"))

        btn_layout.addWidget(self.btn_map, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.btn_full_map, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.btn_open_stl, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.btn_settings, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.btn_exit, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(btn_container)
        layout.addStretch()

        # Conexiones de botones
        self.btn_map.clicked.connect(self._go_map)
        self.btn_full_map.clicked.connect(self._generate_full_map)
        self.btn_full_map.hide()
        self.btn_open_stl.clicked.connect(self._open_stl)
        self.btn_settings.clicked.connect(self._settings)
        self.btn_exit.clicked.connect(self.close)

    def _go_menu(self):
        self.stack.setCurrentWidget(self.menu_page)

    def _go_map(self):
        self.map_page.load()
        self.stack.setCurrentWidget(self.map_page)

    def _open_stl(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Abrir STL", str(Paths.stl_dir), "STL (*.stl)")
        if fn:
            # Mostrar diálogo de carga mientras se abre el modelo
            loading_dialog = ProgressDialog("Cargando modelo STL...")
            loading_dialog.show()

            def open_model():
                loading_dialog.update_progress(50, f"Cargando {os.path.basename(fn)}...")
                self.viewer_page.load_stl(fn)
                self.stack.setCurrentWidget(self.viewer_page)
                loading_dialog.close()

            # Usar QTimer para permitir que se muestre el diálogo antes de cargar
            QTimer.singleShot(300, open_model)

    def _settings(self):
        # Configuración de carpetas
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Configuración")
        settings_dialog.setMinimumWidth(500)

        layout = QVBoxLayout(settings_dialog)

        # HGT Directory
        hgt_layout = QHBoxLayout()
        hgt_label = QLabel("Carpeta de archivos HGT:")
        hgt_path = QLabel(str(Paths.hgt_dir))
        hgt_path.setStyleSheet("font-weight: bold;")
        hgt_btn = QPushButton("Cambiar...")

        hgt_layout.addWidget(hgt_label)
        hgt_layout.addWidget(hgt_path, 1)  # 1 = stretch factor
        hgt_layout.addWidget(hgt_btn)

        # STL Directory
        stl_layout = QHBoxLayout()
        stl_label = QLabel("Carpeta de modelos STL:")
        stl_path = QLabel(str(Paths.stl_dir))
        stl_path.setStyleSheet("font-weight: bold;")
        stl_btn = QPushButton("Cambiar...")

        stl_layout.addWidget(stl_label)
        stl_layout.addWidget(stl_path, 1)  # 1 = stretch factor
        stl_layout.addWidget(stl_btn)

        # Close button
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(settings_dialog.accept)

        layout.addLayout(hgt_layout)
        layout.addLayout(stl_layout)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Connect buttons
        def change_hgt():
            new_dir = QFileDialog.getExistingDirectory(settings_dialog, "Seleccionar carpeta de HGT", str(Paths.hgt_dir))
            if new_dir:
                # Actualizar la ruta utilizando el método de Paths
                Paths.update_hgt_dir(new_dir)
                self.map_page.hgt_root = str(Paths.hgt_dir)
                hgt_path.setText(str(Paths.hgt_dir))
                QMessageBox.information(settings_dialog, "Configuración actualizada",
                                       f"Carpeta HGT configurada en:\n{Paths.hgt_dir}")

        def change_stl():
            new_dir = QFileDialog.getExistingDirectory(settings_dialog, "Seleccionar carpeta de STL", str(Paths.stl_dir))
            if new_dir:
                # Actualizar la ruta utilizando el método de Paths
                Paths.update_stl_dir(new_dir)
                stl_path.setText(str(Paths.stl_dir))
                QMessageBox.information(settings_dialog, "Configuración actualizada",
                                       f"Carpeta STL configurada en:\n{Paths.stl_dir}")

        hgt_btn.clicked.connect(change_hgt)
        stl_btn.clicked.connect(change_stl)

        settings_dialog.exec()

    def _on_tile_selected(self, hgt_path: str):
        # 1. Volver al menú principal y mostrar progreso
        self.stack.setCurrentWidget(self.menu_page)

        # 2. Mostrar diálogo de progreso con estilo mejorado
        self.progress_dlg = ProgressDialog("Generando modelo 3D...")
        self.progress_dlg.setWindowTitle(f"Generando modelo de {os.path.basename(hgt_path)}")
        self.progress_dlg.show()

        # 3. Iniciar worker en segundo plano
        self.worker = STLWorker(hgt_path)
        self.worker.progress.connect(self.progress_dlg.update_progress)
        self.worker.finished.connect(self._on_generation_finished)
        self.worker.start()

    def _on_generation_finished(self, success: bool, stl_path: str, err: str):
        if self.progress_dlg:
            # Mostrar finalización en la barra de progreso
            if success:
                self.progress_dlg.update_progress(100, "¡Modelo generado correctamente!")
                # Breve pausa para mostrar el 100% antes de cerrar
                QTimer.singleShot(800, self.progress_dlg.close)
            else:
                self.progress_dlg.close()
                self.progress_dlg = None

        if not success:
            QMessageBox.critical(self, "Error", f"Ocurrió un error generando el STL:\n{err}")
            self._go_menu()
            return

        # 4. Mostrar modelo 3D
        self.viewer_page.load_stl(stl_path)
        self.stack.setCurrentWidget(self.viewer_page)

    def _generate_full_map(self):
        # Crear un diálogo para seleccionar opciones
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("Opciones de Generación de Mapa")
        options_dialog.setMinimumWidth(400)

        layout = QVBoxLayout(options_dialog)

        # Título
        title = QLabel("Selecciona las opciones para generar el mapa:")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)

        # Opciones de área
        area_group = QGroupBox("Área a generar:")
        area_layout = QVBoxLayout()

        ecuador_radio = QRadioButton("Solo Ecuador (más rápido)")
        ecuador_radio.setChecked(True)
        all_radio = QRadioButton("Todos los archivos HGT disponibles")

        area_layout.addWidget(ecuador_radio)
        area_layout.addWidget(all_radio)
        area_group.setLayout(area_layout)

        # Opciones de resolución
        resolution_group = QGroupBox("Resolución:")
        resolution_layout = QVBoxLayout()

        low_radio = QRadioButton("Baja (más rápido, archivo pequeño)")
        medium_radio = QRadioButton("Media (equilibrado)")
        high_radio = QRadioButton("Alta (más detalle, archivo grande)")

        medium_radio.setChecked(True)

        resolution_layout.addWidget(low_radio)
        resolution_layout.addWidget(medium_radio)
        resolution_layout.addWidget(high_radio)
        resolution_group.setLayout(resolution_layout)

        # Botones
        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Cancelar")
        generate_btn = QPushButton("Generar")
        generate_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")

        buttons.addWidget(cancel_btn)
        buttons.addWidget(generate_btn)

        # Añadir todo al layout principal
        layout.addWidget(area_group)
        layout.addWidget(resolution_group)
        layout.addLayout(buttons)

        # Conexiones
        cancel_btn.clicked.connect(options_dialog.reject)
        generate_btn.clicked.connect(options_dialog.accept)

        # Mostrar diálogo
        if options_dialog.exec() == QDialog.DialogCode.Accepted:
            # Obtener opciones seleccionadas
            only_ecuador = ecuador_radio.isChecked()

            if low_radio.isChecked():
                resolution = "low"
            elif high_radio.isChecked():
                resolution = "high"
            else:
                resolution = "medium"

            # 1. Volver al menú principal y mostrar progreso
            self.stack.setCurrentWidget(self.menu_page)

            # 2. Mostrar diálogo de progreso para la generación del mapa
            area_text = "Ecuador" if only_ecuador else "todos los archivos HGT"
            title_text = f"Generando mapa 3D de {area_text}"

            self.progress_dlg = ProgressDialog(title_text)
            self.progress_dlg.setWindowTitle(title_text)
            self.progress_dlg.show()

            # 3. Iniciar worker en segundo plano para la generación del mapa
            self.worker = STLWorker(generate_full_map=True, only_ecuador=only_ecuador, resolution=resolution)
            self.worker.progress.connect(self.progress_dlg.update_progress)
            self.worker.finished.connect(self._on_full_map_generation_finished)
            self.worker.start()
        else:
            # Cancelado
            return

    def _on_full_map_generation_finished(self, success: bool, stl_path: str, err: str):
        if self.progress_dlg:
            # Mostrar finalización en la barra de progreso
            if success:
                self.progress_dlg.update_progress(100, "¡Mapa generado correctamente!")
                # Breve pausa para mostrar el 100% antes de cerrar
                QTimer.singleShot(800, self.progress_dlg.close)
            else:
                self.progress_dlg.close()
                self.progress_dlg = None

        if not success:
            QMessageBox.critical(self, "Error", f"Ocurrió un error generando el mapa STL:\n{err}")
            self._go_menu()
            return

        # 4. Mostrar modelo 3D del mapa generado
        self.viewer_page.load_stl(stl_path)
        self.stack.setCurrentWidget(self.viewer_page)
