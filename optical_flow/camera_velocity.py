import cv2
import numpy as np
import time
import sys

cap = cv2.VideoCapture(0)  # Using a camera instead of a video file
ret, frame1 = cap.read()

if not ret:
    print("Video cannot be loaded.")
    sys.exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

times = []

while True:
    start_time = time.time()  # Start of timing

    ret, frame2 = cap.read()
    if not ret:
        print("Video clip has finished")
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.3, 2, 10, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Contour detection based on flow magnitude
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 400:  # Variable contour size criterion
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            speed_text = 'v=' + str(int(np.mean(hsv[y:y+h, x:x+w, 2])))  # Approximate velocity
            cv2.putText(frame2, speed_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    end_time = time.time()  # End of timing
    times.append(end_time - start_time)

    cv2.imshow('frame', frame2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('optical_fb.png', frame2)
        cv2.imwrite('optical_hsv.png', bgr)

    prvs = next_frame

cap.release()
cv2.destroyAllWindows()

if times:
    print(f"Average frame processing time: {sum(times) / len(times):.2f} seconds.")