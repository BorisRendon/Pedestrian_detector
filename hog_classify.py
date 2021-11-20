from joblib import dump, load
import cv2 as cv
import img_utils as t   
import hog_model_train as hog_train 
import numpy as np
import pandas as pd
import sys
#cargar modelo
clf = load('SVM_Og.joblib') 

#cargar imagen

variable = sys.argv[1]
print(variable)

imagen = cv.imread(f"test_eval/{variable}")
img2 = cv.cvtColor(imagen,cv.COLOR_BGR2RGB)

##### NOTA #######
#Si quisieran generar el .csv desde 0 de nuevo y persistir el modelo, lo pueden hacer con la siguiente linea de codigo:
#hog_train.write_csv() #por default genera un csv con el nombre que puse en la función, basta con cambiar el nombre ahi dentro para generar el csv con otro nombre
#hog_train.persist_model("Pedestrians_OG1.csv") #por default genera un archivo .joblib co el nombre que puse en la función, basta con cambiar el nombre ahí dentro para generar el joblib con otro nombre
#y debemos de cambiar la línea 9 e este file para leer el otro modelo

#correr estos das funciones toma aproximadamente unos 5-6 minutos (dependiendo la computadora) en mi máquina (macbook pro chip M1)
#### NOTA ####


def encabezado():
    '''
    Creamos un arreglo de 3780 posiciones y lo convertimos a string para ponerlo como encabezado del dataframe

    Returns:
        Lista (np.array): Lista con valores strings de 0 a 3779
    
    '''
    enc = np.arange(3780)
    lista = []
    for i in enc :
        lista.append(str(i))
    return np.array(lista)



c = [] #coordenadas persona
c2=[]
encabs = encabezado()
for i in range(128, len(img2)-128, 128): #poder cambiar 
    for j in range(64, len(img2[0])-64, 64): 
        celda = img2[i-128:i, j-64:j] #puedo cambiar para probar otra ventana
       
        vector_hog = hog_train.histograma_norm(hog_train.arreglo_histogramas(celda)).reshape(1,3780)
        muestra = pd.DataFrame(vector_hog, columns = encabs)
        prueba = clf.predict(muestra.values)
        if prueba[0] == 1: 
            c.append(f"{i-128},{j-64}") 

#aqui dibujo el recuadro verde
esquinas = list(dict.fromkeys(c))
for i in range(128, len(img2), 128):
    for j in range(64, len(img2[0]), 64):
        if f"{i},{j}" in esquinas:
            cv.rectangle(img2, (i, j), (i+64, j+128), (0,255,0), 2)

#para otro size

titulo = sys.argv[2]
output = sys.argv[3]
t.imgview(img2,titulo,output)

if __name__ == "__main__":
  pass