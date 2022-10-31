import cv2 as cv
import prepro as pp
import numpy as np

def kmeans(img):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def contorno(img):
    #img=pp.prePro(img)
    contorno=pp.edgeDetection(img)
    return contorno

def mediana(img):
    median=cv.medianBlur(img,5)
    return median

def binario(img):
    #img=pp.prePro(img)
    (T, bin) = cv.threshold(img, 0, 255,cv.THRESH_OTSU)
    return bin


def prewtxy(img):
    #gray=pp.prePro(img)
    prewittx=np.array([[-1, -1,  -1],
                   [0,  0, 0],
                    [1, 1,  1]])

    prewitty=np.array([[-1, 0,  1],
                   [-1,  0, 1],
                    [-1, 0,  1]])
 
    filtradax=pp.filtro(prewittx,img)
    
    filtraday=pp.filtro(prewitty,img)

    filtradaxy = cv.addWeighted(filtradax, 0.5, filtraday, 0.5, 0)
    return filtradaxy

def main():  
    # define a video capture object
    vid = cv.VideoCapture(0)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        frame=pp.resized(frame,150)

       #frame=pp.prePro(frame)

        #frame=binario(frame)

        img=kmeans(frame)


        # Display the resulting frame
        cv.imshow("camera",img)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()

main()