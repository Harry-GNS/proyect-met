import tkinter as tk
from tkinter import filedialog
import os

def select_hgt_file(initial_dir='data'):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Selecciona un archivo HGT',
        initialdir=initial_dir,
        filetypes=[('HGT files', '*.hgt')]
    )
    return file_path if file_path else None
