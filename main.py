from hgt_map_selector import select_hgt_by_map
from terrain import process_hgt

if __name__ == '__main__':
    file_path = select_hgt_by_map('Datos')
    if not file_path:
        print('No se seleccionó ningún archivo.')
    else:
        print(f'Archivo seleccionado: {file_path}')
        process_hgt(file_path)
