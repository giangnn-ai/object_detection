import numpy as np
import time
import cv2
import os
import sys
#import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment
from PIL import ImageFont, ImageDraw, Image

AudioSegment.converter = "D:/Giang/DL/object_detection/ffmpeg/bin/ffmpeg.exe"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('./video.avi',fourcc, 30.0, (640,480))
#coding=UTF8
# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names", encoding="utf8").read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

while True:
	frame_count += 1
    # Capture frame-by-frameq
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	frames.append(frame)

	if ret:
		key = cv2.waitKey(1)
		if frame_count % 90 == 0:
			end = time.time()
			# grab the frame dimensions and convert it to a blob
			(H, W) = frame.shape[:2]
			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []
			centers = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > 0.5:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))

			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

			texts = []

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the frame
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						confidences[i])
					
					#cv2.putText(frame, text, (x, y - 5),
					#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					
					b,g,r,a = 0,255,0,0
					fontpath ='D:/Giang/DL/object_detection/font/kosugi.ttf' # Windows10 だと C:\Windows\Fonts\ 以下にフォントがあります。
					font = ImageFont.truetype(fontpath, 20) # フォントサイズが32

					img_pil = Image.fromarray(np.asarray(frame)) # 配列の各値を8bit(1byte)整数型(0～255)をPIL Imageに変換。

					draw = ImageDraw.Draw(img_pil) # drawインスタンスを生成

					position = (x, y - 20) # テキスト表示位置
					draw.text(position, text, font = font , fill = (b, g, r, a) ) # drawにテキストを記載 fill:色 BGRA (RGB)

					frame = np.array(img_pil) # PIL を配列に変換
					
					cv2.imshow('frame',frame)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
					
					# find positions
					centerX, centerY = centers[i][0], centers[i][1]
					
					if centerX <= W/3:
						W_pos = " 左"
					elif centerX <= (W/3 * 2):
						W_pos = ""
					else:
						W_pos = " 右"
					
					if centerY <= H/3:
						H_pos = "上, "
					elif centerY <= (H/3 * 2):
						H_pos = " 真ん中, "
					else:
						H_pos = "下, "

					texts.append(H_pos + W_pos + LABELS[classIDs[i]])

			print_texts = [x.encode('utf-8') for x in texts]
			#print(print_texts)
			
			if texts:
				description = ', '.join(texts)
				tts = gTTS(description, lang='ja')
				tts.save('tts.mp3')
				tts = AudioSegment.from_mp3("tts.mp3")
				subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])
			
		cv2.imshow('frame',frame)
		out.write(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break	
cap.release()
out.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")