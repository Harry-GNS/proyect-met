from hgt_map_selector import select_hgt_by_map
import tkinter as tk
from tkinter import messagebox
import multiprocessing as mp
import sys

def main():
    while True:
        file_path = select_hgt_by_map('Datos')
        if not file_path:
            print('No se seleccionó ningún archivo.')
            break
        print(f'Archivo seleccionado: {file_path}')
        # Procesar usando la lógica optimizada de Mapa3DPrint.py
        # Ejecutar el script como módulo para asegurar compatibilidad con multiprocessing
        import subprocess
        result = subprocess.run([sys.executable, 'Mapa3DPrint.py', file_path])
        root = tk.Tk()
        root.withdraw()
        resp = messagebox.askyesno('Elegir otro archivo', '¿Quieres elegir otro archivo HGT?')
        root.destroy()
        if not resp:
            break

if __name__ == '__main__':
    mp.freeze_support()
    main()
