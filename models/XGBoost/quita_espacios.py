#!/usr/bin/env python3
"""
quita_espacios.py — elimina los espacios de los nombres de archivo
en el directorio donde se ejecuta.

- Reemplaza cada espacio `" "` por nada (`""`).
- No toca subcarpetas.
- Si el nuevo nombre ya existe, añade un sufijo numérico para evitar colisiones.
"""

import os
from pathlib import Path

def quitar_espacios():
    cwd = Path.cwd()

    for path in cwd.iterdir():
        if path.is_file() and " " in path.name:
            nuevo_nombre = path.name.replace(" ", "")
            nuevo_path   = path.with_name(nuevo_nombre)

            # Evitar sobreescribir si ya existe un archivo con ese nombre
            contador = 1
            while nuevo_path.exists():
                nuevo_path = path.with_name(f"{nuevo_path.stem}_{contador}{nuevo_path.suffix}")
                contador += 1

            print(f"Renombrando: {path.name}  →  {nuevo_path.name}")
            path.rename(nuevo_path)

if __name__ == "__main__":
    quitar_espacios()
