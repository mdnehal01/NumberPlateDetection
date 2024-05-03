import cv2

cap = cv2.VideoCapture("Video/demo.mp4")

NumberPlateCascade = cv2.CascadeClassifier("cascades/haarcascade_russian_plate_number.xml")

count = 0

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    if success:

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        numberPlate = NumberPlateCascade.detectMultiScale(frame_gray, 1.1, 10)

        for x,y,w,h in numberPlate:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 2)
            cv2.putText(frame, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, ( 0,222,0), 2)
            frameROI = frame[y:y+h, x:x+w]

        cv2.imshow("PLATE", frameROI)
        cv2.imshow("Out", frame)

        if cv2.waitKey(50) & 0xFF == ord("1"):
            cv2.imwrite("Resources/NumberPlate/No_plate"+str(count)+".jpg", frameROI)
            cv2.rectangle(frame, (0,200), (640, 300), (0,255,0), cv2.FILLED)
            cv2.putText(frame, "Scan Saved", (150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,21), 2)
            cv2.imshow("Out video", frame)
            cv2.waitKey(500)
            count += 1
            
    else:
        break

cap.release()
cv2.destroyAllWindows()
