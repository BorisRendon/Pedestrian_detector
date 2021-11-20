# -*- coding: utf-8 -*-

#Boris Rendon - 20180497
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

#1. imgview

def imgview(img,title=None,filename=None):
  """
  Muestra color o escala de grises.

  args:
    img (np.array): imagen a mostrar
    title (str): El tito de la imagen (opcional)
    filename (str): Nombre para guardar la imagen en formato .png (opcional)

  Returns:
    Visualizacion de la imagen
  """

  #Primer paso, verificar los argumentos opcionales:
    #titulo
  if title != None:
    plt.title(title)


  plt.axis('off')
  if len(img.shape) >2:
    plt.imshow((img).astype(np.uint8), vmin=0, vmax=255)
    
  else:
    plt.imshow((img).astype(np.uint8), vmin=0, vmax=255,cmap='gray')
    
    #filename
  if filename != None:
    plt.savefig(filename +".png")

# 2. imgcmp

def imgcmp(img1,img2,title=None,filename=None):
  """
  Muestra dos imagenes para poderlas comparar lado a lado

  Args:
    img1 (np.array): Imagen a mostrar 1
    img2 (np.array): Imagen a mostrar 2
    title (lista): Titulo para cada imagen (opcional)
    filename (str): Nombre para guardar la imagen en formato .png (opcional)

  Returns:
    Visualizacion de la imagen
  """

  #inicializo los titulos de las imagens como None:

  titles=[None,None]

  if title != None:
    titles=title
  

  fig=plt.figure(figsize=(15,15))

  ax1=fig.add_subplot(221)
  imgview(img1,titles[0])
  ax1.axis('off')

  #imagen 2

  ax2=fig.add_subplot(222)
  imgview(img2,titles[1])
  ax2.axis('off')

  if filename != None:
    plt.savefig(filename +".jpg")

#parte 2 img_sobel


def img_kernel(img):
  """
  Calculamos la magnitud y direccion del gradiente Sobel

  Args:
    img (np.array): Una imagen a blanco y negro

  Returns:
    Visualizacion de la magnitud y direccion del gradiente de la imagen.
  """
  img_=np.pad(img,2,'constant')
  sx=cv.Sobel(img_,cv.CV_32F,1,0,ksize=1)
  sy=cv.Sobel(img_,cv.CV_32F,0,1,ksize=1)

  magnitud= np.sqrt((sx**2)+(sy**2))
  direccion = np.rad2deg(np.arctan2(sx,sy))%180 # convertir los angulos de radianes a grados en mod 180


  magnitud = magnitud[2:-2,2:-2]#quito el padding de la imagen que le puse arriba
  direccion = direccion[2:-2,2:-2]##quito el padding de la imagen que le puse arriba

  return magnitud,direccion


if __name__ == "__main__":
  pass