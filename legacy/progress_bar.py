import tkinter as tk
from tkinter import ttk

class ProgressBarWindow:
    def __init__(self, max_value, title='Procesando...'):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('400x100')
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=350, mode='determinate', maximum=max_value)
        self.progress.pack(pady=30)
        self.label = tk.Label(self.root, text='0%')
        self.label.pack()
        self.root.update()
    def update(self, value):
        self.progress['value'] = value
        percent = int((value / self.progress['maximum']) * 100)
        self.label.config(text=f'{percent}%')
        self.root.update()
    def close(self):
        self.root.destroy()
