import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# Defininimos función para mostrar imágenes
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

# -----------------------------------------------------------------------------------
# --- Medidas Invariantes de Momentos: Momentos de Hu--------------------------------
# -----------------------------------------------------------------------------------

# --- Cargo imagen ---------------------------------------
img = cv2.imread("avion_01.png", cv2.IMREAD_GRAYSCALE)
print(img.shape)
imshow(img)

# --- Medidas invariantes de Momentos --------------------
moments = cv2.moments(img)        # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga556a180f43cab22649c23ada36a8a139
type(moments)
len(moments)
moments

for k, v in moments.items():
    # print(k, v)
    # print(f"{k:4}: {v}")
    # print(f"{k:4}: {v:5.2e}")
    print(f"{k:4}: {v:+5.2e}")

# --- Momentos de Hu ------------------------------------
# En base a dichas medidas invariantes de los momentos, obtenemos los Momentos de Hu
hu_moments = cv2.HuMoments(moments)
type(hu_moments)
hu_moments.shape
hu_moments

hu_moments = hu_moments.flatten()   # Notar arriba que originalmente es una matriz de dimensiones (7,1) --> pasamos a vector (7,)
for ii,vv in enumerate(hu_moments):
    print(f"{ii:1d}: {vv:5.2e}")

# -----------------------------------------------------------------------------------------------------------
# --- Ejemplo 1: Analisis de un objeto en diferentes perspectivas -------------------------------------------
# -----------------------------------------------------------------------------------------------------------
img = cv2.imread("aviones.png", cv2.IMREAD_GRAYSCALE)
imshow(img, title="Imagen Original")
# La idea es obtener cada imagen de avión por separado, calcular los momentos de Hu de cada uno, y luego compararlos.

# --- Segmentamos cada avión ------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
print(num_labels)
print(stats)
imshow(labels)

# --- Avion 1 ---------------------------------------
r_ini = stats[1, cv2.CC_STAT_TOP]
c_ini = stats[1, cv2.CC_STAT_LEFT]
r_end = stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT]
c_end = stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH]
avion_1 = img[r_ini:r_end, c_ini:c_end]

# --- Avion 2 ---------------------------------------
r_ini = stats[2, cv2.CC_STAT_TOP]
c_ini = stats[2, cv2.CC_STAT_LEFT]
r_end = stats[2, cv2.CC_STAT_TOP] + stats[2, cv2.CC_STAT_HEIGHT]
c_end = stats[2, cv2.CC_STAT_LEFT] + stats[2, cv2.CC_STAT_WIDTH]
avion_2 = img[r_ini:r_end, c_ini:c_end]

# --- Avion 3 ---------------------------------------
r_ini = stats[3, cv2.CC_STAT_TOP]
c_ini = stats[3, cv2.CC_STAT_LEFT]
r_end = stats[3, cv2.CC_STAT_TOP] + stats[3, cv2.CC_STAT_HEIGHT]
c_end = stats[3, cv2.CC_STAT_LEFT] + stats[3, cv2.CC_STAT_WIDTH]
avion_3 = img[r_ini:r_end, c_ini:c_end]

# --- Visualizo -------------------------------------
plt.figure()
plt.subplot(131), imshow(avion_1, new_fig=False, title="Avion 1")
plt.subplot(132), imshow(avion_2, new_fig=False, title="Avion 2")
plt.subplot(133), imshow(avion_3, new_fig=False, title="Avion 3")
plt.show(block=False)

# --- Obtengo descriptores ---------------------------------------
# Ahora, calculamos los momentos de Hu para cada avión, y los 
# disponemos en una única matriz de resultados para facilitar la comparación
avion_1_HM = cv2.HuMoments(cv2.moments(avion_1))
avion_2_HM = cv2.HuMoments(cv2.moments(avion_2))
avion_3_HM = cv2.HuMoments(cv2.moments(avion_3))

results = np.zeros((7,3))    # filas: 7 momentos de Hu / columnas: 3 aviones
results[:,0] = avion_1_HM.flatten()
results[:,1] = avion_2_HM.flatten()
results[:,2] = avion_3_HM.flatten()
print(results)


# -------------------------------------------------------------------------------------------------
# --- Ejemplo 2: Misma imagen con diferentes transformaciones -------------------------------------
# -------------------------------------------------------------------------------------------------

# --- Cargo imagen ---------------------------------------
img = cv2.imread("pintura.tif", cv2.IMREAD_GRAYSCALE)
img.shape
W,H = img.shape
imshow(img, title="Imagen Original")

# =========================================================================================================
# Ahora, crearemos varias versiones transformadas de esta imagen (escalado, traslación, rotación, etc.), 
# para evaluar si los momentos de Hu son robustos con respecto a estas transformaciones.
# 
# Para una mejor visualización, todas las imágenes tendrán el mismo tamaño (600x600), para lo cual, una 
# vez transformada la imagen, relllenamos con 0s para llegar a dicho tamaño.
# =========================================================================================================

# --- Imagen 1 - Original ------------------------------------------------
M = 100
img1 = cv2.copyMakeBorder(img, M, M, M, M, cv2.BORDER_CONSTANT, None, 0)
img1.shape
imshow(img1, title="Imagen Original")

# --- Imagen 2 - Escalada 0.5 ---------------------------------------------
aux = cv2.resize(img, (W//2, H//2))
aux.shape
M = 200
img2 = cv2.copyMakeBorder(aux, M, M, M, M, cv2.BORDER_CONSTANT, None, 0)
img2.shape
imshow(img2, title="Imagen escalada 0.5")

# --- Imagen 3 - Escalada 1/8 ---------------------------------------------
aux = cv2.resize(img, (W//8, H//8))
aux.shape
M = 275
img3 = cv2.copyMakeBorder(aux, M, M, M, M, cv2.BORDER_CONSTANT, None, 0)
img3.shape
imshow(img3, title="Imagen escalada 1/8")

# --- Imagen 4 - Rotada 15 ------------------------------------------------
aux = imutils.rotate_bound(img, 15)
aux.shape
img4 = cv2.copyMakeBorder(aux, 55, 56, 55, 56, cv2.BORDER_CONSTANT, None, 0)
img4.shape
imshow(img4, title="Imagen rotada 15°")

# --- Imagen 5 - Rotada -45 -------------------------------------------
aux = imutils.rotate_bound(img, -45)
aux.shape
img5 = cv2.copyMakeBorder(aux, 17, 18, 17, 18, cv2.BORDER_CONSTANT, None, 0)
img5.shape
imshow(img5, title="Imagen rotada -45°")

# --- Imagen 6 - Rotada 195 ---------------------------------------------
aux = imutils.rotate_bound(img, 195)
aux.shape
img6 = cv2.copyMakeBorder(aux, 55, 56, 55, 56, cv2.BORDER_CONSTANT, None, 0)
img6.shape
imshow(img6, title="Imagen rotada 195°")

# --- Imagen 7 - Escaclada 0.5 y rotada 115 -----------------------------
aux = cv2.resize(img, (W//2, H//2))
aux = imutils.rotate_bound(aux, 115)
aux.shape
img7 = cv2.copyMakeBorder(aux, 167, 168, 167, 168, cv2.BORDER_CONSTANT, None, 0)
img7.shape
imshow(img7, title="Imagen escalada 0.5 + rotada 115°")

# --- Imagen 8 - Escaclada 1/8, rotada -35 y trasladada  ----------------
aux = cv2.resize(img, (W//8, H//8))
aux = imutils.rotate_bound(aux, -35)
aux.shape
img8 = cv2.copyMakeBorder(aux, 100, 431, 100, 431, cv2.BORDER_CONSTANT, None, 0)
img8.shape
imshow(img8, title="Imagen escalada 1/8, rotada 15° y trasladada")

# --- Calculo los momentos de Hu ----------------------------------------
hu1 = cv2.HuMoments(cv2.moments(img1))
hu2 = cv2.HuMoments(cv2.moments(img2))
hu3 = cv2.HuMoments(cv2.moments(img3))
hu4 = cv2.HuMoments(cv2.moments(img4))
hu5 = cv2.HuMoments(cv2.moments(img5))
hu6 = cv2.HuMoments(cv2.moments(img6))
hu7 = cv2.HuMoments(cv2.moments(img7))
hu8 = cv2.HuMoments(cv2.moments(img8))

x = np.concatenate((hu1, hu2, hu3, hu4, hu5, hu6, hu7, hu8), axis=1)
x.shape
x
print(x)
with np.printoptions(suppress=False, linewidth=100, precision=2):
    print(x)

# --- Análisis ---------------------------------------------------------------------------------------
mean = np.mean(x, axis=1)
std = np.std(x, axis=1)
mean.shape
std.shape

x_stats = np.array([mean,std]).T
print(x_stats)

# ===================================================================================================
# Si observamos la media (m) y la desviación standard (std) de cada momento de Hu, podemos observar 
# que la std es siempre un orden de magnitud menor como mínimo respecto a la m, llegando a ser 
# hasta 4 órdenes menor.
# 
# Esto indica que todos los resultados son muy parecidos en este caso.
# 
# Como conclusión, se puede obvservar que los momentos de Hu son "invariantes" frente a estas 
# transformaciones.
# ===================================================================================================
