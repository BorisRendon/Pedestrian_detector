import numpy as np
import img_utils as utils
import csv
import cv2 as cv
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd 



def magnitud_direction(celda):
    '''
    Calculamos el valor máximo y su dirección respectiva

    Args:
        celda (numpy array): una matriz de tamaño 8x8x3 (que corresponde a un pixlel de cada canal)
    Returns:
        magnitud,direction (tupla): Arreglo con los valores máximos de la magnitud y la dirección que le corresponde. 
    '''

    Rmag,Rdir = utils.img_kernel(celda[:,:,0])
    Bmag,Bdir = utils.img_kernel(celda[:,:,1])
    Gmag,Gdir = utils.img_kernel(celda[:,:,2])
    #obtemenos la magintud y la dirección de cada canal (RGB)

    direccion = np.zeros((8, 8))
    magnitud=np.zeros((8,8))

    
    #instanciamos una matriz de tamaño 8x8 para almacenar las direcciones  

    for i in range(celda.shape[0]): #recorremos el for en ambas dimensiones
        for j in range(celda.shape[1]):#recorremos el for en ambas dimensiones
            arr = [Rmag[i,j] , Bmag[i,j] ,Gmag[i,j] ] #Almacenamos el valor que corresponde a la magnitud
            dir = [Rdir[i,j] , Bdir[i,j] , Gdir[i,j]] #almacenamos el indice que corresponde a la direccion
            maximo =max(enumerate(arr),key=lambda x: x[1])[0]
            direccion[i,j]= dir[maximo] #obtenemos el valor maximo de dir , que corresponde a la direccion
            magnitud[i,j] = arr[maximo] #btenemos el valor maximo de arr , que corresponde a la magnitud

    return magnitud ,direccion

def histograma(direccion, magnitud):

    '''
    Calculamos el HOG de para una celda de tamaño 8x8

    Args:
        direccion (numoy array) = Arreglo con valores de direccion.
        magnitud (numpy array) = Arreglo con valores de magnitud.
    Returns:
        celda (numpy array)

    '''
    celda = np.zeros(9) #en este arreglo vamos a almecenar el histograma
    angulos = [0,20,40,60,80,100,120,140,160] #armamos el arreglo de los ángulos donde vamos a crear nuestro histograma, 0 también es = a 180

    for i in range(len(celda)  -1): #tomamos incrementos de 8 en 8 porque así podemos consumir toda la celda en una sola corrida en el primer eje
        for j in range (len(celda) -1): #también tomamos incrementos de 8 en 8 en el segundo eje
            #hacemos la primera verificación si el ángulo es menor o igual a 0.
            if direccion[i,j] <=0: #Caso base
                celda[0]+= magnitud[i,j]
            elif direccion[i,j]>160: #caso donde el ángulo se encuentre entre 160 y 180/0
                celda[-1] += magnitud[i,j] * (1-(abs(180-direccion[i,j])/20)) # lo que hacemos aqui es hacemos la resta de 1-el % que es la distancia entre el número y el limite superior (180)
                celda[0] += magnitud[i,j] * (1-(abs(160-direccion[i,j])/20))
            
            #ahora falta el caso donde son valores mas grandes de 0 pero más pequeños que 160

            else:
                for u in range(len(angulos)-1):
                    if direccion[i,j] > angulos[u] and direccion[i,j] <= angulos[u+1]:
                        celda[u+1]+= magnitud[i,j] * (1-(abs(angulos[u+1] - direccion[i,j])/20)) #almacenamos el valor en el angulo mayor
                        celda[u]+= magnitud[i,j] * (1-(abs(angulos[u] - direccion[i,j])/20)) #almacenamos el valor en el angulo menor
                        
    

    return celda


def arreglo_histogramas(img):
    '''
    Armamos el arreglo de los histogramas
    Args:
        img: recibimos una imagen para sacarle la magnitud y dirección correspondiente
    Returns:
        hist_arr = Devolvemos el histograma de la imagen
    '''

    hist_arr= np.zeros((16,8,9)) #lo armamos de este tamaño porque son 16 columnas, 8 filas y en cada una tenemos el histograma que corresponde.

    for i in range(8,len(img)+8,8): #lo hacemos de esta manera porque al recorrer de 8 en 8 podemos ir de celda por celda en las filas
        for j in range(8,len(img[0])+8,8): #también lo hacemos de 8 en 8 para poder ahora ir columna por columna
            magnitud , direccion = magnitud_direction(img[i-8:i,j-8:j])
            hist_arr[i//8 -1 , j//8 -1 ] = histograma(direccion,magnitud)

    return hist_arr

def histograma_norm(histogramas):
    '''
    Normalizamos el bloque
    Args:
        histogramas (numpy array): arreglo de histogramas
    Returns:
        hist_norm (numpy array) : Devolvemos el histograma normalizado en el shape que necesitamos(3780)
    
    '''
    hist_norm = np.zeros((15,7,36)) #tenemos que armar un arreglo de este tamaño porque son los bloques que podemos obtener en una imagen.
    for i in range (2,len(histogramas)+1):
        for j in range(2,len(histogramas[0])+1):
            bloque = histogramas[i-2:i, j-2:j , : ].reshape(36)
            if bloque.sum() <=0:
                hist_norm[i-2,j-2] = 0
            else:
                hist_norm[i-2,j-2] = bloque / np.sqrt(np.sum(bloque**2)) #fómula de normalización del bloque.
    
    #arr_hist = arreglo_histogramas(img)

    return hist_norm.reshape(3780) 

def write_csv():
    '''
    Escribimos un csv el dataset que creamos y le ponemos un
    feature extra a los que tienen una persona 1 y un 0 a los que no
    '''
    folder = 'dataset/Pedestrians-Dataset/Pedestrians'
    folder2 = 'dataset/Pedestrians-Dataset/Background'

    f = open('Pedestrians_OG1.csv' , 'w')
    writer = csv.writer(f)
    for filename in os.listdir(folder) :
            if filename.endswith(".png"):
                img_1 = cv.imread(folder + '/' + filename)
                
                img_2 = cv.cvtColor(img_1,cv.COLOR_BGR2RGB)
                histograma_ =arreglo_histogramas(img_2)
                histograma_normalizado = histograma_norm(histograma_)
                pedestrians = np.hstack((histograma_normalizado,[1]))

                writer.writerow(pedestrians)

    for filename in os.listdir(folder2) :
            if filename.endswith(".png"):
                img_1 = cv.imread(folder2 + '/' + filename)
                
                img_2 = cv.cvtColor(img_1,cv.COLOR_BGR2RGB)
                histograma_ =arreglo_histogramas(img_2)
                histograma_normalizado = histograma_norm(histograma_)
                pedestrians = np.hstack((histograma_normalizado,[0]))

                writer.writerow(pedestrians)

    f.close    


def persist_model(csv_file):
    '''
    En esta función entrenamos el modelo y lo persistimos en un archivo .joblib
    Documentacion de la persistencia del modelo : https://scikit-learn.org/stable/modules/model_persistence.html
    Args:
        csv_file(archivo de extensión .csv) : recibimos el csv con el modelo


    '''
    file_csv= pd.read_csv(csv_file, header = None)

    y=file_csv[3780]
    x=file_csv.iloc[:,:-1]
    X_train, X_test, y_train, y_test = train_test_split(x ,y, test_size=0.30, random_state=42)
    clf = SVC(kernel='linear')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    model_score = accuracy_score(y_test,y_pred)
    print(f'El acurracy del modelo es de: {model_score}')

    dump(clf,'SVM_Og1.joblib')



