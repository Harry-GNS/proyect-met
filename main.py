from hgt_map_selector import select_hgt_by_map
from terrain import process_hgt
import tkinter as tk
from tkinter import messagebox

def main():
    while True:
        file_path = select_hgt_by_map('Datos')
        if not file_path:
            print('No se seleccionó ningún archivo.')
            break
        print(f'Archivo seleccionado: {file_path}')
        process_hgt(file_path)
        root = tk.Tk()
        root.withdraw()
        resp = messagebox.askyesno('Elegir otro archivo', '¿Quieres elegir otro archivo HGT?')
        root.destroy()
        if not resp:
            break

if __name__ == '__main__':
    main()
