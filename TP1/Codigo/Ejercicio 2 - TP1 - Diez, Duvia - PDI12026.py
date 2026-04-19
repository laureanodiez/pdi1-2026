def detectar_lineas_tabla(img_bin):
    """
    Recibe una imagen binarizada (fondo 0, líneas 1 o 255) y devuelve las 
    posiciones de las líneas divisorias horizontales y verticales.
    """
    # 1. Sumar filas y columnas
    # img_cols = np.sum(..., axis=0)
    # img_rows = np.sum(..., axis=1)
    
    # 2. Definir umbrales y encontrar picos (las líneas)
    # --- TU TAREA AQUÍ ---
    
    return [], [] # Retornar listas o arrays con coordenadas X e Y de las líneas

def limpiar_celda_y_extraer_letra(celda_img):
    """
    Procesa el recorte de una sola celda.
    Utiliza cv2.connectedComponentsWithStats para aislar la letra y eliminar ruido.
    """
    # 1. Aplicar componentes conectadas
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(...)
    
    # 2. Filtrar por área (th_area) para eliminar restos de bordes o ruido
    # --- TU TAREA AQUÍ ---
    
    # 3. Analizar la componente restante para saber si es A, B, C o D.
    
    return "A" # Letra detectada (o None si está vacía/nula)

def procesar_examen(ruta_imagen):
    """
    Función principal (Orquestador)
    """
    # 1. Cargar la imagen
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # 2. Binarización
    # Tip: Para sumar píxeles fácilmente, conviene que las líneas negras sean 1 y el fondo blanco 0.
    # _, img_bin = cv2.threshold(...)
    
    # 3. Detectar estructura (líneas)
    # lineas_horiz, lineas_vert = detectar_lineas_tabla(img_bin)
    
    # 4. Recortar celdas del encabezado y validarlas
    # ...
    
    # 5. Recortar celdas de respuestas y analizarlas
    # ...
    
    # 6. Comparar con las respuestas correctas dadas en el enunciado
    # ...
    
    return resultados