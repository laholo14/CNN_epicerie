import cv2
import tensorflow as tf
import numpy as np
import time
import os



PRODUIT = ['Gouty','Le Fruit','Siligaoma','Tic Tac']
COLEUR_PRODUIT = np.random.uniform(0, 255, size=(len(PRODUIT), 3))


def prepare(image):
    tab_image  = cv2.imread(os.path.join(image))
    taille_resize = cv2.resize(tab_image,(160, 160))
    image = np.expand_dims(taille_resize,axis=0)
    return image

epicerie = tf.keras.models.load_model("model\epicerie.h5")


num_image = 0
couleur = (255, 0, 255)
label = '...'
camera = cv2.VideoCapture(1)

while True:
    _,image = camera.read()


    image = cv2.resize(image,(800,600))
    blur = cv2.Laplacian(image, cv2.CV_64F).var()
    
    position = (630, 550)
    
    floutage = cv2.Laplacian(image, cv2.CV_64F).var()

    if  cv2.waitKey(1)%256 == 32 : 

        nowTime = time.time()        
        nom_image = "capture/"+ str(num_image) + "_" + str(nowTime)+ ".png"

        cv2.imwrite(nom_image, image)
        num_image += 1

        produit_prediction = epicerie.predict(prepare(nom_image)).argmax()
        label = PRODUIT[produit_prediction]
        couleur = COLEUR_PRODUIT[produit_prediction]
        print(label)
    
    elif cv2.waitKey(1)%256 == 27:
        break

    
    if True :
        cv2.putText(image,label , position, cv2.FONT_HERSHEY_SIMPLEX, 1,couleur , 3)
        cv2.imshow("Reconnaissance de Produit application 1", image)


    
camera.release()
