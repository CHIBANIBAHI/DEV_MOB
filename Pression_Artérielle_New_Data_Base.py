# Load libraries
import cv2
import numpy as np
import scipy.signal as sig
from scipy.interpolate import interp1d
#from pushbullet import Pushbullet


faceDetect=cv2.CascadeClassifier('/home/giveit/Bureau/DDDEEEVVV/projet mobile/haarcascade_frontalface_alt.xml')
#cam = cv2.VideoCapture(0)
#img=cv2.imread('/home/giveit/Bureau/DDDEEEVVV/projet mobile/reconnaissance/image_10.jpg')
id=1; # changer pour chaque nouvelle personne
sampleNum=0
while(True):
	#ret, img=cam.read()
	img=cv2.imread('/home/giveit/Bureau/DDDEEEVVV/projet mobile/reconnaissance/image_177.jpg')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceDetect.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		sampleNum=sampleNum+1
		cv2.imwrite("/home/giveit/Bureau/DDDEEEVVV/projet mobile/PHOTO_21_03//photos."+str(id)+'.'+ str(sampleNum) + ".jpg",
gray[y:y+h,x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.waitKey(100);
	cv2.imshow("reconnaissance faciale",img)
	cv2.waitKey(1000);
	if (sampleNum>1):
	    break
#cam.release()
#cv2.destroyAllWindows()

#photos/User
#15

#########################################################
# Extract PPG signal
signal = gray.mean(axis=0)
fs = 30  # Sampling rate in Hz
f, Pxx = sig.periodogram(signal, fs, scaling='density')
hrv = f[np.argmax(Pxx)]
t = np.arange(len(signal)) / fs
interp_signal = interp1d(t, signal)


# Estimate blood pressure
systolic_amp = interp_signal(0.4 / hrv) - interp_signal(0.25 / hrv)
diastolic_amp = interp_signal(0.7 / hrv) - interp_signal(0.4 / hrv)
systolic_bp = 120 + 6.5 * systolic_amp
diastolic_bp = 80 + 4.5 * diastolic_amp
mean_bp = (systolic_bp + 2 * diastolic_bp) / 3

print("Systolic blood pressure: ", systolic_bp)
print("Diastolic blood pressure: ", diastolic_bp)
print("Mean blood pressure: ", mean_bp)

#######################################"

