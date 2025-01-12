import os
import cv2
import csv
import numpy as np
from feat import Detector

detector = Detector(
    face_model = "retinaface",
    landmark_model = "mobilefacenet",
    au_model = "svm",
    emotion_model = "resmasknet",
)

cap = cv2.VideoCapture(0)

auLabels = [
    'AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU11','AU12',
    'AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28','AU43'
]

selectedAUs = ['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU11','AU12',
'AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28','AU43']

selected = [auLabels.index(au) for au in selectedAUs]

auPred = []

while True:
   ret, frame = cap.read()
   if not ret:
      break

   frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
   try:
      facePred = detector.detect_faces(frameRGB)
      landmarkPred = detector.detect_landmarks(frameRGB, facePred)
      
      if landmarkPred is not None:
         auPredictions = detector.detect_aus(frameRGB, landmarks=landmarkPred)
         print(f"AU predictions structure: {auPredictions}")
         
         for idx, auValues in enumerate(auPredictions):
               print(f"Face {idx + 1} AU predictions: {auValues}")

               flattened = auValues.flatten().tolist()
               filtered = [flattened[i] for i in selected]
               
               auPred.append(filtered)
         
         if facePred is not None:
               for i, box in enumerate(facePred):
                  if isinstance(box, list) and len(box) == 4:
                     x1, y1, x2, y2 = [int(coord) for coord in box]
                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                     
                     if i < len(auPred):
                           prominentAuIndex = np.argmax(auPred[i])
                           prominentAuLabel = selected[prominentAuIndex]
                           cv2.putText(frame, prominentAuLabel, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
   except Exception as e:
      print(f"Error in detection: {e}")
      continue

   cv2.imshow('Live AU Detection', frame)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

outputFile = "AUdata.csv"
if auPred:
    print(f"Exporting filtered AU data to {outputFile}")
    
    with open(outputFile, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(selectedAUs)
        
        writer.writerows(auPred)
    
    print(f"Filtered AU data successfully exported to {outputFile}")
else:
    print("No AU data to export.")
