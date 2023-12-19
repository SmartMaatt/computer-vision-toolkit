import cv2
import numpy as np
from PIL import ImageGrab


clip_path = 'clip.mp4'
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(clip_path)

frame_count = 0
previous_frame = None
prepared_frame = None
ret, frame = cap.read()

while True:
    frame_count += 1
    ret, img_rgb = cap.read()
    if not ret:
      print('No frames grabbed!')
      break

    if previous_frame is None:
      previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      hsv = np.zeros_like(frame)
      hsv[..., 1] = 255
      continue

    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous_frame, prepared_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('prepared_frame', bgr)
    k = cv2.waitKey(30) & 0xff

    if k == 27:
      break
    elif k == ord('s'):
      cv2.imwrite('opticalfb.png', img_rgb)
      cv2.imwrite('opticalhsv.png', bgr)
    previous_frame = prepared_frame

    # hey_as_long_as_it_works.jpg
    for h in range (0, 255):
      maskHSV = cv2.inRange(bgr, np.array([h,0,10]), np.array([h,255,255]))
      contours, _ = cv2.findContours(image=maskHSV, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
     
      if (not contours):
        continue

      if cv2.contourArea(contours[0]) < 50:
        # too small: skip!
        continue
     
      (x, y, w, h) = cv2.boundingRect(contours[0])
      cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
      cv2.putText(img_rgb,'v=' + str(bgr[int(y + h * 0.5)][int(x + w * 0.5)][2]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)

    #bgr_g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    #contours, _ = cv2.findContours(image=bgr_g, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      if cv2.contourArea(contour) < 400:
        # too small: skip!
        continue
     
      (x, y, w, h) = cv2.boundingRect(contour)
      cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
      cv2.putText(img_rgb,'v=' + str(hsv[int(y + h * 0.5)][int(x + w * 0.5)][2]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
      print("Countour pixel: [" +
      str(hsv[int(y + h * 0.5)][int(x + w * 0.5)][0]) +
      "] [" +
      str(hsv[int(y + h * 0.5)][int(x + w * 0.5)][1]) +
      "] [" +
      str(hsv[int(y + h * 0.5)][int(x + w * 0.5)][2]) +
      "]")
    frame = img_rgb

    cv2.imshow('frame', frame) # press escape to exit
    if (cv2.waitKey(30) == 27):
      break

cap.release()
cv2.destroyAllWindows()