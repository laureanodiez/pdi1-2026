import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ===============================================================================
# Función imshow de la cátedra
# ===============================================================================
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# ===============================================================================
# --- Problema - Cinco dados ----------------------------------------------------
# ===============================================================================

videos = sorted(glob.glob('TP3/videos/tirada_*.mp4'))
if not videos:
    videos = sorted(glob.glob('tirada_*.mp4'))
    
if not videos:
    print("No se encontraron videos con el formato 'tirada_<id>.mp4'.")

output_dir = os.path.join('TP3', 'resultados')
os.makedirs(output_dir, exist_ok=True)

tirada_nro = 1  

for video_path in videos:
    video_name = os.path.basename(video_path)
    print(f"\n{'='*50}\nProcesando: {video_name}\n{'='*50}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video {video_name}.")
        continue
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    nombre_salida = os.path.join(output_dir, f'Resultado_{video_name}')
    out = cv2.VideoWriter(nombre_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Variables de estado de la máquina
    estado = 'BUSCANDO'
    centroides_anteriores = None
    contador_frames_quietos = 0
    
    dados_reposo = [] 
    resultados_tirada = [] 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_result = frame.copy()
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # --- 1) SEGMENTACIÓN DE LOS DADOS ---
        mask_verde = cv2.inRange(frame_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask_sin_verde = cv2.bitwise_not(mask_verde)

        mask_rojo1 = cv2.inRange(frame_hsv, np.array([0,  100, 60]), np.array([10,  255, 255]))
        mask_rojo2 = cv2.inRange(frame_hsv, np.array([165, 100, 60]), np.array([180, 255, 255]))
        mask_rojo = cv2.bitwise_or(mask_rojo1, mask_rojo2)
        
        mask_dados = cv2.bitwise_and(mask_rojo, mask_sin_verde)
        
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_dados = cv2.morphologyEx(mask_dados, cv2.MORPH_CLOSE, kernel_c)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dados, connectivity=8, ltype=cv2.CV_32S)
        
        dados_actuales = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            aspect_ratio = float(w) / h
            
            if 800 < area < 20000 and 0.5 < aspect_ratio < 1.8:
                dados_actuales.append({'bbox': (x, y, w, h), 'centroide': centroids[i]})

        # --- 2) LÓGICA DE MOVIMIENTO ---
        if len(dados_actuales) == 5:
            dados_actuales = sorted(dados_actuales, key=lambda d: d['centroide'][0])
            centroides_actuales = np.array([d['centroide'] for d in dados_actuales])
            
            if centroides_anteriores is not None:
                distancias = np.sqrt(np.sum((centroides_actuales - centroides_anteriores)**2, axis=1))
                if np.all(distancias < 5.0): 
                    contador_frames_quietos += 1
                else:
                    contador_frames_quietos = 0
                    estado = 'BUSCANDO' 
                    
            centroides_anteriores = centroides_actuales
            
            # --- 3) TRANSICIÓN AL REPOSO Y CÁLCULO DE PUNTOS ---
            if contador_frames_quietos > 15 and estado == 'BUSCANDO':
                estado = 'REPOSO'
                dados_reposo = dados_actuales
                
                resultados_tirada = []
                rois_bgr_list = []
                rois_bin_list = []
                
                for k in range(5):
                    x, y, w, h = dados_reposo[k]['bbox']
                    roi_dado_bgr = frame[y:y+h, x:x+w]
                    roi_dado = frame_gris[y:y+h, x:x+w]
                    
                    k_size = max(3, int(min(w, h) * 0.20)) 
                    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                    roi_tophat = cv2.morphologyEx(roi_dado, cv2.MORPH_TOPHAT, kernel_tophat)
                    
                    _, roi_bin = cv2.threshold(roi_tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Imágenes para el informe
                    #if tirada_nro == 1 and k == 1:
                        #cv2.imwrite("TP3/resultados/1_dado_gris_traslucido.jpg", roi_dado)
                        #cv2.imwrite("TP3/resultados/2_dado_tophat.jpg", roi_tophat)
                        #cv2.imwrite("TP3/resultados/3_dado_otsu_limpio.jpg", roi_bin)

                    rois_bgr_list.append(roi_dado_bgr)
                    rois_bin_list.append(roi_bin)
                    
                    c_num, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(roi_bin, connectivity=8, ltype=cv2.CV_32S)
                    
                    puntos_detectados = 0
                    area_dado = w * h
                    
                    for j in range(1, c_num):
                        c_w = c_stats[j, cv2.CC_STAT_WIDTH]
                        c_h = c_stats[j, cv2.CC_STAT_HEIGHT]
                        c_area = c_stats[j, cv2.CC_STAT_AREA]
                        c_ar = float(c_w) / c_h
                        
                        # FIX: Cota inferior subida a 0.015 para descartar reflejos del flash
                        if (area_dado * 0.015) < c_area < (area_dado * 0.15) and 0.4 < c_ar < 2.5:
                            puntos_detectados += 1
                            
                    resultados_tirada.append(puntos_detectados)

                # ===============================================================
                # PASO 5: REPORTE POR CONSOLA Y CUMPLIMIENTO DEL AVISO
                # ===============================================================
                
                # A. Reporte por consola ordenado
                tiempo_seg = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                print(f"\n  TIRADA {tirada_nro}  |  t={tiempo_seg:.2f}s")
                for i, puntos in enumerate(resultados_tirada):
                    print(f"    Dado {i+1}: {puntos} puntos")
                print(f"    SUMA TOTAL: {sum(resultados_tirada)}\n")

                # Actualizamos frame_result para que la captura lo muestre con las cajas
                for idx, dado in enumerate(dados_reposo):
                    x, y, w, h = dado['bbox']
                    cv2.rectangle(frame_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_result, f"D{idx+1}-{resultados_tirada[idx]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # B. Gráfica de Matplotlib (Se muestra 2.5 segs y sigue el video)
                plt.figure(figsize=(12, 7))
                plt.suptitle(f"Tirada {tirada_nro} - Suma Total: {sum(resultados_tirada)}", fontsize=16, fontweight='bold')
                
                plt.subplot(2, 2, 1)
                plt.imshow(cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB))
                plt.title("1. Video en reposo (Frame completo)")
                plt.axis('off')
                
                plt.subplot(2, 2, 2)
                plt.imshow(mask_dados, cmap='gray')
                plt.title("2. Capa imagen de detección (HSV)")
                plt.axis('off')
                
                plt.subplot(2, 1, 2)
                dice_visuals = []
                for i in range(5):
                    bgr_res = cv2.resize(rois_bgr_list[i], (90, 90))
                    bin_res = cv2.cvtColor(cv2.resize(rois_bin_list[i], (90, 90)), cv2.COLOR_GRAY2BGR)
                    
                    lbl = np.zeros((35, 90, 3), dtype=np.uint8)
                    cv2.putText(lbl, f"D{i+1}: {resultados_tirada[i]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    col = np.vstack([lbl, bgr_res, bin_res])
                    dice_visuals.append(col)
                    
                    if i < 4: 
                        dice_visuals.append(np.zeros((215, 20, 3), dtype=np.uint8))
                        
                final_row = np.hstack(dice_visuals)
                plt.imshow(cv2.cvtColor(final_row, cv2.COLOR_BGR2RGB))
                plt.title("3. Dados cropeados con sus resultados (Original y TopHat+Otsu)")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show(block=True)
                
                tirada_nro += 1
                
        else:
            contador_frames_quietos = 0
            centroides_anteriores = None
            # No cambiamos el estado acá, para que los recuadros se mantengan hasta que los muevan físicamente.
            
        # Actualización de estados si se vuelve a tirar un dado en el mismo video
        if estado == 'REPOSO':
            if len(dados_actuales) == 5:
                centroides_act = np.array([d['centroide'] for d in dados_actuales])
                centroides_rep = np.array([d['centroide'] for d in dados_reposo])
                if np.max(np.sqrt(np.sum((centroides_act - centroides_rep)**2, axis=1))) > 20.0:
                    estado = 'BUSCANDO'
                    contador_frames_quietos = 0
            
            # --- 4) DIBUJO EN PANTALLA CONSTANTE DURANTE EL REPOSO ---
            for idx, dado in enumerate(dados_reposo):
                x, y, w, h = dado['bbox']
                cv2.rectangle(frame_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_result, f"D{idx+1}-{resultados_tirada[idx]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
        out.write(frame_result)

    cap.release()
    out.release()
    print(f"Video guardado exitosamente en: {nombre_salida}")

cv2.destroyAllWindows()