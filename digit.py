import numpy as np
import cv2 as cv
#import tensorflow as tf

#sign = tf.keras.datasets.mnist
#(x_train, y_train),(x_test,y_test) = sign.load_data()


#m_new = tf.keras.models.load_model('model_mnist.h5')


wname = 'Digits'
drawing = False

img = np.zeros([400,400], dtype = 'uint8')*255
cv.namedWindow(wname)

def shape(event,x,y,flags,param):
    global drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing ==True:
            cv.circle(img,(x,y),10,(255,0,0),-1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.circle(img,(x,y),10,(255,0,0),-1)

cv.setMouseCallback(wname, shape)
while True:
    cv.imshow(wname, img)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        out =img[50:350,50:350]
        cv.imwrite('Output.jpg', out)
        out_resize = cv.resize(out,(28,28)).reshape(1,28,28)
        #m_new.predict_classes(out_resize)
    elif key == ord('n'):
        img[30:400,30:400]=0
cv.destroyAllWindows()
