FROM python:3.10.12

RUN apt update -y
RUN pip install --upgrade pip
 
RUN pip install opencv-python-headless numpy imageio scikit-image pycocotools

# If needed for opencv
RUN apt-get install -y libgl1-mesa-dev 


COPY . .

# ENTRYPOINT [ "python", "/main.py" ]
ENTRYPOINT [ "python", "/resize.py" ]
# ENTRYPOINT [ "python", "/showAnnotation.py"]




