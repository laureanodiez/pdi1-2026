import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local_histograma(img, M, N):
    """
    Aplica ecualización local del histograma a una imagen en escala de grises.
    M, N: Dimensiones de la ventana (alto, ancho). ¡Deben ser números impares!
    """
    # 1. Obtener dimensiones de la imagen original
    alto, ancho = img.shape
    
    # 2. Calcular el "padding" (cuántos píxeles extra necesitamos en los bordes)
    pad_y = M // 2
    pad_x = N // 2
    
    # 3. Agregar bordes a la imagen usando replicación, como sugiere el TP
    img_padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    
    # 4. Crear una imagen negra (ceros) del mismo tamaño original para guardar el resultado
    img_out = np.zeros_like(img)
    
    # 5. Recorrer la imagen original píxel a píxel
    for i in range(alto):
        for j in range(ancho):
            # --- TU TAREA AQUÍ ---
            # a. Recortar la "ventana" de tamaño MxN de img_padded.
            #    Ten cuidado con los índices: el centro de esta ventana corresponde a (i + pad_y, j + pad_x)
            # ventana = img_padded[..., ...]
            
            # b. Aplicar cv2.equalizeHist() a esa ventana recortada.
            # ventana_ecualizada = ...
            
            # c. Extraer únicamente el valor del píxel central de la ventana ecualizada
            #    y guardarlo en img_out[i, j].
            pass # Elimina este pass cuando escribas tu código
            
    return img_out

# Código de prueba (para cuando completes la función)
# img_oculta = cv2.imread('Imagen_con_objetos_ocultos.tiff', cv2.IMREAD_GRAYSCALE)
# resultado = ecualizacion_local_histograma(img_oculta, 15, 15)
# plt.imshow(resultado, cmap='gray')
# plt.show()