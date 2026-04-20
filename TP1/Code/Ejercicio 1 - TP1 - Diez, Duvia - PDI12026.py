import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta = r'pdi1-2026\TP1\Images\Imagen_con_detalles_escondidos.tif'
img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)


def ecualizacion_local(img, M, N):
    """
    Realiza la ecualización local del histograma.
    M: Alto de la ventana (filas)
    N: Ancho de la ventana (columnas)
    """
    # Calculamos cuánto borde necesitamos agregar (la mitad de la ventana)
    pad_y = M // 2
    pad_x = N // 2
    
    # Agregamos el borde replicando el último píxel
    img_pad = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    
    # Creamos una imagen vacía del mismo tamaño que la original para guardar el resultado
    img_out = np.zeros_like(img)
    
    filas, columnas = img.shape
    
    # Movemos el centro de la ventana de píxel en píxel
    for i in range(filas):
        for j in range(columnas):
            # Extraemos la sub-imagen (la ventana M x N)
            ventana = img_pad[i : i + M, j : j + N]
            
            # Ecualizamos SOLO esa pequeña ventana
            ventana_ecualizada = cv2.equalizeHist(ventana)
            
            # Mapeamos el nivel de intensidad del píxel centrado en la ventana
            # y lo guardamos en nuestra imagen de salida
            img_out[i, j] = ventana_ecualizada[pad_y, pad_x]
           
    return img_out


if img is None:
    print("Error: No se encontró la imagen.")
else:
    # Definimos el tamaño de la ventana
    M, N = 30, 30 
    
    print(f"Procesando ecualización local con ventana {M}x{N}... esto puede tardar un momento.")
    
    # Aplicamos la función 
    img_heq_local = ecualizacion_local(img, M, N)

    # Ploteo
    plt.figure(figsize=(12, 10))
     
    ax1 = plt.subplot(221)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Original')
    
    plt.subplot(222)
    plt.hist(img.flatten(), 256, [0, 256])
    plt.title('Histograma Global Original')
    
    plt.subplot(223, sharex=ax1, sharey=ax1)
    plt.imshow(img_heq_local, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Ecualización Local (Ventana {M}x{N})')
    
    plt.subplot(224)
    plt.hist(img_heq_local.flatten(), 256, [0, 256])
    plt.title('Histograma Post-Ecualización Local')
    
    plt.tight_layout()
    plt.show()

    


def calcular_cdf(imagen):
    hist, _ = np.histogram(imagen.flatten(), 256, [0, 256])
    histn = hist.astype(np.double) / imagen.size
    return histn.cumsum()

# Calculamos los 3 CDFs 
cdf_original = calcular_cdf(img)
#cdf_global = calcular_cdf()           # Imagen de cv2.equalizeHist()
cdf_local = calcular_cdf(img_heq_local)      # Imagen de tu función local

# Graficamos los tres juntos para comparar
plt.figure(figsize=(8, 6))
plt.plot(cdf_original, color='blue', label='CDF Original')
#plt.plot(cdf_global, color='green', label='CDF Global (cv2)')
plt.plot(cdf_local, color='red', label='CDF Local')

plt.title('Comparación de Funciones de Distribución Acumulada (CDF)')
plt.legend(loc='upper left')
plt.show()


"""
Análisis:
    Al delimitar la ecualización a una ventana
    hace que el modelo no generalice la distribución 
    de grises agarrando el fondo liso dentro de la ecuación.

    Mientras mas chica es la ventana menos generalizacion hay,
    haciendo que haya cada vez mas contraste y ruido, en cambio 
    si agrandamos la ventana la imagen se ve mas natural y estable 
    pero perdemos detalles
"""