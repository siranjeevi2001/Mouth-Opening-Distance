import cv2
import numpy as np

face_cascade_path = 'xlm files/haarcascade_frontalface_default.xml'
mouth_cascade_path = 'xlm files/haarcascade_mouth.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

if face_cascade.empty() or mouth_cascade.empty():
    print("Error loading cascade classifiers")
def detect_face_and_mouth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    face_coordinates_list = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 15))

        for (mx, my, mw, mh) in mouths:
            face_coordinates_list.append((x, y, w, h, mx, my, mw, mh))

    return face_coordinates_list

def calculate_mouth_opening_distance(face_coordinates_list):
    distances = []

    for face_coordinates in face_coordinates_list:
        if face_coordinates is not None:
            face_x, face_y, face_w, face_h, mx, my, mw, mh = face_coordinates

            # Calculate mouth opening distance (you can customize this based on your needs)
            distance = abs((my + mh // 2) - (face_y + face_h // 2))
            distances.append(distance)

    return distances

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        face_coordinates_list = detect_face_and_mouth(frame)

        # Calculate and display mouth opening distance for each face
        distances = calculate_mouth_opening_distance(face_coordinates_list)
        for i, distance in enumerate(distances):
            cv2.putText(frame, f'Mouth Opening Distance {i + 1}: {distance}', (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Mouth Opening Distance Assessment', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
