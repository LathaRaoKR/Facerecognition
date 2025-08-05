import cv2
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace

# === Step 1: Load names from Training_images ===
path = 'Training_images'
classNames = []
myList = os.listdir(path)

for cl in myList:
    classNames.append(os.path.splitext(cl)[0])
print("Registered Faces:", classNames)


# === Step 2: Attendance Marking ===
def markAttendance(name):
    filename = 'Attendance.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Time\n")

    df = pd.read_csv(filename)
    if name not in df['Name'].values:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        with open(filename, 'a') as f:
            f.write(f'\n{name},{dtString}')
        print(f"{name} marked present at {dtString}")


# === Step 3: Start Webcam ===
cap = cv2.VideoCapture(0)
print("Starting camera...")

while True:
    success, img = cap.read()
    if not success:
        break

    try:
        # === Step 4: Recognize Face ===
        results = DeepFace.find(img_path=img, db_path=path, enforce_detection=False)

        if isinstance(results, list) and len(results) > 0:
            df = results[0]
            if not df.empty:
                best_match_path = df.iloc[0]['identity']
                name = os.path.splitext(os.path.basename(best_match_path))[0]
                markAttendance(name)

                # Draw rectangle and name on face
                cv2.putText(img, name.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Face not recognized:", e)

    cv2.imshow("Webcam Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
