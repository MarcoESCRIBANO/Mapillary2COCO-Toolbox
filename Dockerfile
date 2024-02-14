# FROM docker-base-mapillary-dataset as build-env
FROM python:3.10.12

# COPY --from=build-env MapillaryVistas MapillaryVistas
RUN apt update -y
RUN pip install --upgrade pip
 
RUN pip install opencv-python-headless numpy imageio scikit-image pycocotools
RUN apt-get install -y libgl1-mesa-dev


COPY . .



ENTRYPOINT [ "python", "/main.py" ]
# ENTRYPOINT [ "python", "/resize.py" ]
# ENTRYPOINT [ "python", "/ShowMapAnnotation.py"]




