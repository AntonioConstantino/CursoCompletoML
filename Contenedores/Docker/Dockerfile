# Bash es un lenguaje de instrucciones que me permite moverme dentro de mi computadora 

# Usa una imagen base de Python
# Descargo python 
FROM python:3.12 

# Crea un directorio (carpeta) de trabajo
WORKDIR /app

# Copia archivos al contenedor
# copia todos los archivos dentro de la carpeta app
COPY . /app

# Instala dependencias (si tienes un requirements.txt)
RUN pip install -r requirements.txt

# Comando por defecto al ejecutar el contenedor
CMD ["python", "hola_mundo.py"]
