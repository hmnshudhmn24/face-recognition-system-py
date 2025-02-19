import cv2
import face_recognition
import numpy as np
import os

# Load known faces from a folder
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist. Please create it and add images.")
        return [], []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

# Recognize faces from webcam with additional logging and enhancements
def recognize_faces(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)
    frame_count = 0
    face_log = {}

    while True:
        ret, frame = video_capture.read()
        frame_count += 1
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_log[name] = face_log.get(name, 0) + 1

            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            print(f"Detected {name} - Frame Count: {frame_count} - Recognition Count: {face_log.get(name, 0)}")

        cv2.imshow('Face Recognition System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting face recognition system...")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Final Recognition Log:", face_log)

if __name__ == "__main__":
    known_face_encodings, known_face_names = load_known_faces("known_faces")
    if known_face_encodings:
        recognize_faces(known_face_encodings, known_face_names)
    else:
        print("No known faces found in the directory. Please add some images of known persons.")
