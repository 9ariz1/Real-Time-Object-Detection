import cv2
thres=0.5 			# Thresold to detect object
cap=cv2.VideoCapture(0)		# Start video capture using your sysem default camera
cap.set(3,648)			# 3 represent the width with 648px
cap.set(4,448)			# 4 represent the height with 448px
cap.set(10,70)			# 10 represent the brightness with 70%

# class name loading
className=[] #Empty list 
classFile='coco.names'
with open(classFile,'rt') as f:
	className=f.read().rstrip('\n').split('\n')
configPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  	# its contain the feature
weightPath="frozen_inference_graph.pb"				# its contain the model
net=cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320) 				# Resize image to 320x320 px
net.setInputScale(1.0/127.5)				# Normalize px values to [-1 ,1]
net.setInputMean((127.5,127.5,127.5))			# Mean subtraction , subtract 127.5 from each channel
net.setInputSwapRB(True)				# swap channel from BGR to RGB

while True : 
	success,img=cap.read()
	classIds,confs,bbox=net.detect(img,confThreshold=thres)
	if len(classIds) !=0:
		for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
			cv2.rectangle(img,box,color=(0,255,),thickness=2)
			cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
	cv2.imshow("Output",img)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
