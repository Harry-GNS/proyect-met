from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PyQt6.QtCore import Qt, QTimer

class ProgressDialog(QDialog):
    def __init__(self, title="Procesando..."):
        super().__init__()
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(480, 180)

        layout = QVBoxLayout(self)

        # Título más grande y visible
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Etiqueta de estado
        self.label = QLabel("Iniciando proceso...")
        self.label.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self.label)

        # Barra de progreso más grande
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setMinimumHeight(25)
        layout.addWidget(self.bar)

        # Botón de cancelar
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setFixedWidth(100)
        layout.addWidget(self.cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Para animación en caso de que no haya actualizaciones
        self.timer = QTimer()
        self.timer.timeout.connect(self._pulse)
        self.timer.start(150)

    def _pulse(self):
        # Solo anima si no hay progreso definido (valor = 0)
        if self.bar.value() == 0:
            v = (self.bar.value() + 1) % 100
            self.bar.setValue(v)

    def update_progress(self, val: int, text: str = ""):
        # Detener animación si recibimos valor real
        if val > 0:
            self.timer.stop()

        self.bar.setValue(val)
        if text:
            self.label.setText(text)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
