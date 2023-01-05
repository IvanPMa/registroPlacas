#SISTEMAS EXPERTOS - IVAN PACHECO MALDOANDO
import cv2
import numpy as np
import pytesseract
from PIL import Image

#Importados el video
cap = cv2.VideoCapture("videoPlacas.pm4")

Ctexto = ''

#Comanzamos con nuestro while True
while True:
    # Lectura del video
    ret, frame = cap.read()

    if ret == False:
        break

    #Dibujamos el area donde vamos a tomar la muestra
    cv2.rectangle(frame, (870, 750), (1070,850), (0,0,0), cv2.FILLED)
    cv2.putText(frame, Ctexto[0:7], (900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #Extraemos el ancho y el alto de los fotogramas del video
    al, an, c = frame.shape

    #Tomamos el centro de la imagen 
    #En X:
    x1 = int(an / 3) #Tomamos 1/3 de la imagen
    x2 = int(x1 * 2) #Hasta el inicio del 3/3 de la imagen

    # En y:
    y1 = int(an / 3) #Tomamos el 1/3 de la imagen
    y2 = int(y1 * 2) #Hasta el inicio del 3/3 de la imagen

    #Texto en pantalla
    cv2.rectangle(frame, (x1 + 160, y1 + 500), (1120, 940), (0,0,0), cv2.FILED)
    cv2.putText(frame, 'Procesando Placa', (x1 + 180, y1 + 550 ), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)

    #ubicamos el rectangulo en las zonas extraidas
    cv2.rectangle(frame, (x1, y1) , (x2, y2), (0, 255, 0 ), 2)

    #Realizamos un recorte a nuestra zona de interes
    recorte = frame[y1:y2 , x1: x2]

    # Preprocesamiento de la zona de interes
    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    #Color
    Color = cv2.absdiff(mG, mB)

    #Binarizamos la imagen
    _, umbral = cv2.threshold(Color, 40, 255, cv2.THRESH_BINARY)

    #Extraemos los contornos de la zona seleccionada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_ TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Primero los ordenamos del mas grande al mas pequeño
    contornos = sorted(contornos, key lambda x: cv2.contourArea(x), reverse=True)

    #Dibujamos los contornos extraidos
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500 and area < 5000:
            #Detectamos la placa
            x, y, ancho, alto = cv2.cv2.boundingRect(contorno)

            # Extraemos las coordenadas
            xpi = x + x1            #Coordenada de la placa en X inicial
            ypi = y + y1            #Coordenada de la placa en Y inicial

            xpf = x + ancho + x1    #Coordenada de la placa en X final
            ypf = y + alto + y1     #Coordenada de la placa en Y final 

            #Dibujamos el rectangulo 
            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255,255,0), 2)

            #Extraemos los pixeles
            placa = frame[ypi:ypf, xpi:xpf]

            # Extramios el ancho y el alto de los fotogramas
            alp, amp, cp = placa.shape
            #print(alp, anp)

            #Procesamos los pixeles para extraer los valores de las placas
            Mva = np.zeros((alp, anp))

            #Normalizamos la matrices
            mBp = np.matrix(placa[:, :, 0])
            mGp = np.matrix(placa[:, :, 1])
            mRp = np.matrix(placa[:, :, 2])

            # Creamos una mascara
            