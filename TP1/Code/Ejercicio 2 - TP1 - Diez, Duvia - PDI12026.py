"""
Trabajo Práctico 1 - PDI
Laureano Diez, Uriel Duvia

Abril 2026

Ejercicio 2

Se requiere corregir exámenes de forma automática a través de un script de python, dadas 
las respuestas correctas y los exámenes a corregir en imágenes.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


respuestas_correctas = {1:'C', 2:'B', 3:'A', 4:'D', 5:'B', 6:'B', 7:'A', 8:'B', 9:'D', 10:'D'}
#referencia examen1=1-mal 2-mal 3-mal 4-mal 5-mal 6-mal 7-mal 8-mal 9-mal 10-mal = 0/10
#referencia examen2=1-mal 2-mal 3-mal 4-bien 5-mal 6-bien 7-bien 8-mal 9-mal 10-bien = 4/10
#referencia examen3=1-bien 2-bien 3-bien 4-bien 5-bien 6-bien 7-bien 8-bien 9-bien 10-bien = 10/10
#referencia examen4=1-mal 2-mal 3-mal 4-mal 5-mal 6-mal 7-mal 8-mal 9-mal 10-mal = 0/10
#referencia examen5=1-bien 2-bien 3-bien 4-bien 5-bien 6-bien 7-bien 8-bien 9-bien 10-bien = 10/10

def extraer_lineas(array_bool):
    """ Busca dónde arrancan y terminan las tiras de 1s para sacar el centro de la línea """
    centros = []
    en_linea = False
    inicio = 0
    for i in range(len(array_bool)):
        if array_bool[i] and not en_linea:
            en_linea = True
            inicio = i
        elif not array_bool[i] and en_linea:
            en_linea = False
            fin = i
            centros.append(int((inicio + fin) / 2))
    return centros


# --- Filtros de letras (respuestas) ---
def clasificar_letra(recorte_letra):
    alto, ancho = recorte_letra.shape
    if alto == 0 or ancho == 0: 
        return "ERROR"

    img_1 = (recorte_letra > 127).astype(int)

    tinta_arriba = np.sum(recorte_letra[:alto//2, :])
    tinta_abajo = np.sum(recorte_letra[alto//2:, :])
    tinta_izq = np.sum(recorte_letra[:, :ancho//2])
    tinta_der = np.sum(recorte_letra[:, ancho//2:])
    
    tinta_arriba = max(1, tinta_arriba)
    tinta_der = max(1, tinta_der)
    tinta_izq = max(1, tinta_izq)

    medio_der = img_1[alto//3 : 2*(alto//3), ancho//2:]
    densidad_medio_der = np.sum(medio_der) / max(1, medio_der.size)

    if densidad_medio_der < 0.20:
        return 'C'
        
    elif (tinta_abajo / tinta_arriba) > 1.3:
        return 'A'
        
    else:
        y_inicio_centro = alto // 3
        y_fin_centro = 2 * (alto // 3)
        x_inicio_centro = ancho // 3
        x_fin_centro = 2 * (ancho // 3)
        
        centro_letra = recorte_letra[y_inicio_centro:y_fin_centro, x_inicio_centro:x_fin_centro]
        
        tinta_en_el_centro = np.sum(centro_letra)
        area_del_centro = (y_fin_centro - y_inicio_centro) * (x_fin_centro - x_inicio_centro)
        
        densidad_central = tinta_en_el_centro / max(1, area_del_centro)
        
        # DEBUG: ¡Imprimí este valor para calibrar!
        # print(f"      [DEBUG] Densidad central: {densidad_central:.2f}")
        # LA B TIENE LA RAYA DEL MEDIO -> Densidad central ALTA
        # LA D ESTÁ HUECA EN EL MEDIO -> Densidad central BAJA
        
        if densidad_central > 0.30: 
            return 'B'
        else:
            return 'D'


def validar_campo_texto(celda_bin, tipo_campo):
    """ 
    Usamos Bounding Boxes para medir el espacio
    entre letras y deducir cuántas palabras y caracteres hay.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, 8, cv2.CV_32S)

    letras_stats = []
    # Arrancamos de 1 para ignorar el fondo (etiqueta 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 10: # Ignoramos mugre muy chica
            letras_stats.append(stats[i])

    cantidad_letras = len(letras_stats)

    if cantidad_letras == 0:
        return False, "Vacío"

    if tipo_campo == "CLASS":
        if cantidad_letras == 1:
            return True, "OK"
        return False, f"Tiene {cantidad_letras} chars (Debe ser 1)"

    elif tipo_campo == "DATE":
        if cantidad_letras == 8: 
            return True, "OK"
        return False, f"Tiene {cantidad_letras} chars (Deben ser 8)"

    elif tipo_campo == "NAME":
        if cantidad_letras > 25:
            return False, f"Excede 25 chars ({cantidad_letras})"

        letras_stats = sorted(letras_stats, key=lambda s: s[0])
        espacios_grandes = 0
        
        for i in range(len(letras_stats) - 1):
            fin_letra_actual = letras_stats[i][cv2.CC_STAT_LEFT] + letras_stats[i][cv2.CC_STAT_WIDTH]
            inicio_letra_siguiente = letras_stats[i+1][cv2.CC_STAT_LEFT]
            
            distancia = inicio_letra_siguiente - fin_letra_actual
            
            # Si hay más de 5 píxeles de blanco, decimos que es un espacio de palabra
            if distancia > 5: 
                espacios_grandes += 1

        cantidad_palabras = espacios_grandes + 1

        if cantidad_palabras >= 2:
            return True, f"OK ({cantidad_palabras} palabras)"
        else:
            return False, "Solo 1 palabra (Falta apellido)"



directorio_script = os.path.dirname(os.path.abspath(__file__))
carpeta_imagenes = os.path.join(directorio_script, "..", "Images")
archivos_examenes = glob.glob(os.path.join(carpeta_imagenes, "*.png"))
datos_reporte = []

if not archivos_examenes:
    print("No hay exámenes para corregir.")
else:
    print(f"--- Arrancando corrección de {len(archivos_examenes)} exámenes ---\n")

    for ruta in archivos_examenes:
        nombre_archivo = os.path.basename(ruta)
        print(f"==> Analizando: {nombre_archivo}")
        
        # Lectura anti-tildes en la ruta
        img_array = np.fromfile(ruta, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print("  [ERROR] Falla al leer imagen.")
            continue
            
        # --- 1. Binarización ---
        # Invertimos para que la tinta (negro) sea 1/Blanco y el papel (blanco) sea 0/Negro
        _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        img_unos_ceros = (img_bin / 255).astype(np.uint8) 
        
        # --- 2. Detección de la Grilla ---
        img_cols = np.sum(img_unos_ceros, axis=0) 
        img_rows = np.sum(img_unos_ceros, axis=1) 
        
        img_rows_th = img_rows > (img.shape[1] * 0.4) # Umbral empírico
        img_cols_th = img_cols > (img.shape[0] * 0.4)
        
        lineas_h = extraer_lineas(img_rows_th)
        lineas_v = extraer_lineas(img_cols_th)
        print("Coordenadas Y detectadas (lineas_h):", lineas_h)
        print("Coordenadas X detectadas (lineas_v):", lineas_v)
        
        #plt.imshow(img_bin, cmap='gray'); plt.title("Binarizada"); plt.show() # Para debuggear
        
        # --- 3. Validación de Encabezados ---
        print("  [Encabezado]")
        try:
            celda_name = img_bin[0:lineas_h[0], 55:255]
            val, msj = validar_campo_texto(celda_name, "NAME")
            print(f"    Name : {'CORRECTO' if val else 'MAL'} -> {msj}")
            crop_nombre_gris = img[0:lineas_h[0], 55:255]
            
            celda_date = img_bin[0:lineas_h[0], 301:367]
            val, msj = validar_campo_texto(celda_date, "DATE")
            print(f"    Date : {'CORRECTO' if val else 'MAL'} -> {msj}")
            #plt.imshow(celda_date, cmap='gray')
            #plt.show()

            celda_class = img_bin[0:lineas_h[0], 416:552]
            val, msj = validar_campo_texto(celda_class, "CLASS")
            print(f"    Class: {'CORRECTO' if val else 'MAL'} -> {msj}")
            #plt.imshow(celda_class, cmap='gray')
            #plt.show()
            
        except Exception as e:
            print("    [Warning] Hubo un error al recortar el encabezado.")

            
        # --- 4. Corrección de Preguntas (Método de Anclas) ---
        print("  [Respuestas]")
        
        # Recortamos la zona donde están las preguntas (debajo de la primera línea gruesa)
        zona_preguntas = img_bin[lineas_h[1]:, :] 
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zona_preguntas, 8, cv2.CV_32S)
        
        renglones_respuesta = []
        
        # Filtramos para quedarnos con los renglones de respuestas
        for i in range(1, num_labels):
            ancho = stats[i, cv2.CC_STAT_WIDTH]
            alto = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if ancho < 130 and alto < 4 and (ancho / max(1, alto)) > 15:
                renglones_respuesta.append({
                    'x': stats[i, cv2.CC_STAT_LEFT],
                    'y': stats[i, cv2.CC_STAT_TOP] + lineas_h[1], # Le sumamos el corte de arriba para volver a la coord original
                    'w': ancho,
                    'h': alto
                })
        
        # Validamos cuántos renglones encontró.
        #if len(renglones_respuesta) != 10:
        #    print(f"    [ALERTA] Encontré {len(renglones_respuesta)} espacios de respuesta en vez de 10. ¡Calibrar regla de renglones!")
        else:
            # Ordenamos los renglones. 
            # Como están en 2 columnas, los ordenamos primero por X (columna) y luego por Y (fila).
            # Para separar las 2 columnas, dividimos la hoja por la mitad.
            mitad_hoja = img.shape[1] // 2
            
            columna_1 = sorted([r for r in renglones_respuesta if r['x'] < mitad_hoja], key=lambda r: r['y'])
            columna_2 = sorted([r for r in renglones_respuesta if r['x'] >= mitad_hoja], key=lambda r: r['y'])
            
            renglones_ordenados = columna_1 + columna_2 # Preguntas 1 a 5, luego 6 a 10
            
            puntaje = 0
            
            # Analizamos qué hay ARRIBA de cada renglón
            for num_pregunta, renglon in enumerate(renglones_ordenados, start=1):
                
                # Definimos el "Área de captura". 
                # Queremos mirar justo arriba del renglón.
                # X: arranca en el renglón, W: el ancho del renglón.
                # Y: arranca X píxeles más arriba, H: bajamos hasta tocar el renglón.
                
                alto_captura = 14
                y_inicio = max(0, renglon['y'] - alto_captura)
                y_fin = renglon['y']
                x_inicio = renglon['x']
                x_fin = renglon['x'] + renglon['w']
                
                celda_respuesta = img_bin[y_inicio:y_fin, x_inicio:x_fin]
                #plt.imshow(celda_respuesta, cmap='gray')
                #plt.title(f"Recorte arriba del renglón {num_pregunta}")
                #plt.show()
                # break 
                num_labels_ans, _, stats_ans, _ = cv2.connectedComponentsWithStats(celda_respuesta, 8, cv2.CV_32S)
                
                pedazos_tinta = []
                for i in range(1, num_labels_ans):
                    if stats_ans[i, cv2.CC_STAT_AREA] > 5: 
                        pedazos_tinta.append(stats_ans[i])
                
                estado = ""
                if len(pedazos_tinta) == 0:
                    estado = "BLANCO"
                else:
                    min_x = min(p[cv2.CC_STAT_LEFT] for p in pedazos_tinta)
                    max_x = max(p[cv2.CC_STAT_LEFT] + p[cv2.CC_STAT_WIDTH] for p in pedazos_tinta)
                    
                    min_y = min(p[cv2.CC_STAT_TOP] for p in pedazos_tinta)
                    max_y = max(p[cv2.CC_STAT_TOP] + p[cv2.CC_STAT_HEIGHT] for p in pedazos_tinta)
                    
                    ancho_total = max_x - min_x
                    
                    if ancho_total > 25: 
                        estado = "MÚLTIPLE"
                    else:
                        recorte_letra = celda_respuesta[min_y:max_y, min_x:max_x]
                        
                        estado = clasificar_letra(recorte_letra)
                    
                if num_pregunta in respuestas_correctas:
                    correcta = respuestas_correctas[num_pregunta]
                    if estado == correcta:
                        print(f"    Pregunta {num_pregunta}: OK")
                        puntaje += 1
                    else:
                        print(f"    Pregunta {num_pregunta}: MAL (Marcó {estado}, era {correcta})")
                        
            print(f"  --> NOTA FINAL: {puntaje}/10\n")


        esta_aprobado = puntaje >= 6
        datos_reporte.append({
            'nombre_img': crop_nombre_gris,
            'aprobado': esta_aprobado,
            'nota': puntaje
        })

# Creamos el reporte final y lo guardamos en raíz.
if datos_reporte:
    print("Generando imagen de reporte final...")
    
    alto_bloque = 60
    ancho_reporte = 600
    alto_reporte = len(datos_reporte) * alto_bloque
    
    reporte_final = np.ones((alto_reporte, ancho_reporte), dtype=np.uint8) * 255
    
    for i, alumno in enumerate(datos_reporte):
        y_offset = i * alto_bloque
        
        h_crop, w_crop = alumno['nombre_img'].shape
        h_max = min(h_crop, alto_bloque - 10)
        w_max = min(w_crop, 300)
        
        reporte_final[y_offset + 5 : y_offset + 5 + h_max, 10 : 10 + w_max] = \
            alumno['nombre_img'][:h_max, :w_max]
        
        status_texto = f"NOTA: {alumno['nota']} - {'APROBADO' if alumno['aprobado'] else 'DESAPROBADO'}"
        
        cv2.putText(reporte_final, status_texto, (320, y_offset + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0), 2)
        
        cv2.line(reporte_final, (0, y_offset + alto_bloque - 1), 
                 (ancho_reporte, y_offset + alto_bloque - 1), (150), 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(reporte_final, cmap='gray')
    plt.title("Reporte Final de Exámenes")
    plt.axis('off')
    plt.show()
    
    cv2.imwrite("TP1/Images/Reportes/reporte_final.png", reporte_final)