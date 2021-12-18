import tensorflow as tf
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

PRODUIT = ['Gouty','Le Fruit','Siligaoma','Tic Tac']
COLEUR = np.random.uniform(0, 255, size=(len(PRODUIT), 3))

epicerie_model = tf.keras.models.load_model("model\epicerie.h5")
epicerie_xml = cv2.CascadeClassifier("model\cascade.xml")



cap = VideoStream(src=1).start()

while True:

    frame = cap.read()
    frame = imutils.resize(frame, width=800,height=400)
    
    zone_interet = epicerie_xml.detectMultiScale(frame,1.5,2)

    for (x,y,w,h) in zone_interet:
        
        image=frame[y:y+h,x:x+w]
        image=cv2.resize(image,(160,160),interpolation=cv2.INTER_AREA)


        produit_prediction=epicerie_model.predict(np.array(image).reshape(-1,160,160,3)).argmax()  
        label=PRODUIT[produit_prediction]  
        print(label)

        cv2.rectangle(frame,(x,y),(x+w,y+h),COLEUR[produit_prediction],2)

        label_position=(x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,COLEUR[produit_prediction],2)

    cv2.imshow('Reconnaissance de Produit application 3', frame)

    if cv2.waitKey(1)%256 == 27:
            break

cap.release()
cv2.destroyAllWindows()
