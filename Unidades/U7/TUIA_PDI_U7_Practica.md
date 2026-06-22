Universidad Nacional de Rosario 

Facultad de Ciencias Exactas, Ingeniería y Agrimensura Tecnicatura Universitaria en Inteligencia Artificial Procesamiento de Imágenes I - IA 4.4 

## **PRÁCTICA: Unidad 7 - Descriptores y Clasificación** 

## **Ejercicio 1: Segmentación de datos de un DNI** 

A partir del _dataset_ de imágenes que se muestra en la figura 1 (archivos _dni_<id>.png_ ), se requiere realizar un **análisis automático** mediante técnicas de PDI, sobre cada documento nacional de identidad (DNI). 

El objetivo principal es **detectar y segmentar la información presente en el DNI** que se encuentra en cada imagen, para que los resultados sean parte de algún _dataset_ de un modelo de aprendizaje automático de DNI, rostros y/o números. 

Para ello, considere cada una de las imágenes del _dataset_ provisto y desarrolle un _script_ que itere sobre las mismas y de forma automática, permita: 

1. Segmentar en distintas imágenes los elementos DNI, fotografía y número de DNI, de forma individual. 

2. Segmentar en distintas imágenes cada dígito del número de DNI. 

Figura 1: _Dataset_ de imágenes de un par de DNI. 

**AYUDA:** Para detectar y segmentar la fotografía del DNI, puede utilizar el clasificador _Haar Cascade_ (🔗 opencv_haar_cascade_docs), considerando el modelo pre-entrenado de OpenCV: _haarcascade_frontalface_default.xml_ (🔗 frontalface_default_xml). 

**UNR | TUIA | PDI                                                                                                                                     1** 

