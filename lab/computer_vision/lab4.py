import time

import cv2
import cv2 as cv
import numpy as np

cascade_face_fn = "/Users/stone/PycharmProjects/image_processing/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
cascade_smile_fn = "/Users/stone/PycharmProjects/image_processing/venv/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"
cascade_eye_fn = "/Users/stone/PycharmProjects/image_processing/venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"

def DoViolaJonesImage(fn,
                      cascade_face_fn = "haarcascades/haarcascade_frontalface_default.xml",
                      cascade_eye_fn = "haarcascades/haarcascade_eye.xml",
                      cascade_smile_fn = "haarcascades/haarcascade_smile.xml",
                      fn_out = None, minNeighborsFace = 3):
    # Read an image from file
    I = cv.imread(fn, cv.IMREAD_COLOR)
    if not isinstance(I, np.ndarray) or I.data == None:
        print("Error loading file \"{}\"".format(fn))
        exit()

    # Show image
    cv.imshow("Source", I)

    # Convert to grayscale
    Igray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    Igray = cv.equalizeHist(Igray)

    # Load cascade for face
    cascade_face = cv.CascadeClassifier()
    if not cascade_face.load(cv.samples.findFile(cascade_face_fn)):
        print("DoViolaJonesImage: Error loading face cascade {}".format(cascade_face_fn))
        return

    # Load cascade for eye
    cascade_eye = None
    if cascade_eye_fn != None:
        cascade_eye = cv.CascadeClassifier()
        if not cascade_eye.load(cv.samples.findFile(cascade_eye_fn)):
            print("DoViolaJonesImage: Error loading face cascade {}".format(cascade_eye_fn))
            return

    # Load cascade for smile
    cascade_smile = None
    if cascade_smile_fn != None:
        cascade_smile = cv.CascadeClassifier()
        if not cascade_smile.load(cv.samples.findFile(cascade_smile_fn)):
            print("DoViolaJonesImage: Error loading face cascade {}".format(cascade_smile_fn))
            return

    # Detect faces
    faces = cascade_face.detectMultiScale(Igray)#, minNeighbors = minNeighborsFace)

    # Highlight faces
    Iout = I.copy()
    i = 0
    if faces is None:
        print("Empty")
    for (x, y, w, h) in faces:
        Iout = cv.rectangle(Iout, (x, y, w, h), (0, 255, 255), 1)
        Iface = I[y : y + h, x : x + w]

        # Detect eyes in the top 2/3 of the face image
        if cascade_eye != None:
            Iface_top = Igray[y : y + h * 2 // 3, x : x + w]
            eyes = cascade_eye.detectMultiScale(Iface_top, scaleFactor=1.05)
            for (x2, y2, w2, h2) in eyes:
                Iface = cv.rectangle(Iface, (x2, y2, w2, h2), (147, 20, 255), 1)
                Iout  = cv.rectangle(Iout, (x + x2, y + y2, w2, h2), (147, 20, 255), 1)

        # Detect smile in the bottom 1/3 of the face image
        if cascade_smile != None:
            Iface_bottom = Igray[y + h * 2 // 3 : y + h, x : x + w]
            smiles = cascade_smile.detectMultiScale(Iface_bottom, scaleFactor=1.05, minNeighbors=7)
            for (x2, y2, w2, h2) in smiles:
                Iface = cv.rectangle(Iface, (x2, y2 + h * 2 // 3, w2, h2), (0, 255, 0), 1)
                Iout = cv.rectangle(Iout, (x + x2, y + h * 2 // 3 + y2, w2, h2), (0, 255, 0), 1)

        i = i + 1

    # Display an image
    cv.imshow("Found faces", Iout)
    print("i = ", i)

    # Save to file
    if fn_out != None:
        cv.imwrite(fn_out, Iout)

    cv.waitKey()
    cv.destroyAllWindows()


def DoViolaJonesFrame(frame, cascade_face_fn, cascade_eye_fn, cascade_smile_fn, minNeighborsFace=3):
    Igray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    Igray = cv.equalizeHist(Igray)

    cascade_face = cv.CascadeClassifier()
    if not cascade_face.load(cv.samples.findFile(cascade_face_fn)):
        print("DoViolaJonesImage: Error loading face cascade {}".format(cascade_face_fn))
        return

    cascade_eye = None
    if cascade_eye_fn != None:
        cascade_eye = cv.CascadeClassifier()
        if not cascade_eye.load(cv.samples.findFile(cascade_eye_fn)):
            print("DoViolaJonesImage: Error loading face cascade {}".format(cascade_eye_fn))
            return

    cascade_smile = None
    if cascade_smile_fn != None:
        cascade_smile = cv.CascadeClassifier()
        if not cascade_smile.load(cv.samples.findFile(cascade_smile_fn)):
            print("DoViolaJonesImage: Error loading face cascade {}".format(cascade_smile_fn))
            return

    faces = cascade_face.detectMultiScale(Igray)

    frame_out = frame.copy()
    i = 0
    if faces is None:
        print("Empty")
    for (x, y, w, h) in faces:
        frame_out = cv.rectangle(frame_out, (x, y, w, h), (0, 255, 255), 1)
        frame_face = frame[y: y + h, x: x + w]

        # Detect eyes in the top 2/3 of the face image
        if cascade_eye != None:
            Iface_top = Igray[y: y + h * 2 // 3, x: x + w]
            eyes = cascade_eye.detectMultiScale(Iface_top, scaleFactor=1.05, minNeighbors=3)
            for (x2, y2, w2, h2) in eyes:
                frame_face = cv.rectangle(frame_face, (x2, y2, w2, h2), (147, 20, 255), 1)
                frame_out = cv.rectangle(frame_out, (x + x2, y + y2, w2, h2), (147, 20, 255), 1)

        # Detect smile in the bottom 1/3 of the face image
        if cascade_smile != None:
            Iface_bottom = Igray[y + h * 2 // 3: y + h, x: x + w]
            smiles = cascade_smile.detectMultiScale(Iface_bottom, scaleFactor=1.05, minNeighbors=7)
            for (x2, y2, w2, h2) in smiles:
                frame_face = cv.rectangle(frame_face, (x2, y2 + h * 2 // 3, w2, h2), (0, 255, 0), 1)
                frame_out = cv.rectangle(frame_out, (x + x2, y + h * 2 // 3 + y2, w2, h2), (0, 255, 0), 1)

        i = i + 1

    return frame_out

# Video processing
def DoViolaJonesVideo(fn,
                      cascade_face_fn = "haarcascades/haarcascade_frontalface_default.xml",
                      cascade_eye_fn = "haarcascades/haarcascade_eye.xml",
                      cascade_smile_fn = "haarcascades/haarcascade_smile.xml",
                      fn_out = None, minNeighborsFace = 3):
    capture = cv.VideoCapture(fn)

    frame_height = int(capture.get(4))
    frame_width  = int(capture.get(3))
    FPS = capture.get(cv.CAP_PROP_FPS)

    out = cv.VideoWriter(fn_out, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, (frame_width, frame_height))

    ret, frame = capture.read()
    while ret:
        processed_frame = DoViolaJonesFrame(frame, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)
        milliseconds = capture.get(cv.CAP_PROP_POS_MSEC)
        seconds = milliseconds / 1000
        milliseconds %= 1000
        minutes = 0
        if seconds >= 60:
            minutes = seconds / 60
            seconds = seconds % 60
        print(int(minutes), ":", int(seconds), ":", int(milliseconds))
        # cv2.imshow("video", processed_frame)
        out.write(processed_frame)
        ret, frame = capture.read()

        # if cv.waitKey(20) & 0xff == ord('q'):
        #     break

    capture.release()
    out.release()
    cv.destroyAllWindows()


# Camera processing
def DoViolaJonesCamera(cascade_face_fn = "haarcascades/haarcascade_frontalface_default.xml",
                      cascade_eye_fn = "haarcascades/haarcascade_eye.xml",
                      cascade_smile_fn = "haarcascades/haarcascade_smile.xml",
                      minNeighborsFace = 3):
    capture = cv.VideoCapture(0)
    time.sleep(1)

    ret, frame = capture.read()
    while ret:
        frame = cv.flip(frame, 1)
        processed_frame = DoViolaJonesFrame(frame, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)
        cv.imshow("Camera", processed_frame)
        ret, frame = capture.read()

        if cv.waitKey(20) & 0xff == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()



people_in_ITMO = "/Users/stone/PycharmProjects/image_processing/lab1_python/PeopleInITMO.jpg"
teacher_people_2 = "/Users/stone/PycharmProjects/image_processing/lab1_python/teacher_people_2.png"
teacher_musk = "/Users/stone/PycharmProjects/image_processing/lab1_python/teacher_musk.png"
Stone = "/Users/stone/PycharmProjects/image_processing/lab1_python/Stone.png"
# people_1 = "/Users/stone/PycharmProjects/image_processing/lab1_python/people1.jpeg"
# people_2 = "/Users/stone/PycharmProjects/image_processing/lab1_python/people2.jpeg"
# people_3 = "/Users/stone/PycharmProjects/image_processing/lab1_python/photo1.jpg"
face1 = "/Users/stone/PycharmProjects/image_processing/lab1_python/face1.jpg"
face2 = "/Users/stone/PycharmProjects/image_processing/lab1_python/face2.jpg"
face3 = "/Users/stone/PycharmProjects/image_processing/lab1_python/face3.jpg"
# DoViolaJonesImage(face1, cascade_face_fn=cascade_face_fn, cascade_eye_fn=None, cascade_smile_fn=None)
# DoViolaJonesImage(face2, cascade_face_fn=cascade_face_fn, cascade_eye_fn=None, cascade_smile_fn=None)
# DoViolaJonesImage(face3, cascade_face_fn=cascade_face_fn, cascade_eye_fn=None, cascade_smile_fn=None)
# DoViolaJonesImage(face1, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)
# DoViolaJonesImage(face2, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)
# DoViolaJonesImage(face3, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)
# DoViolaJonesImage(people_1, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)

video = "/Users/stone/PycharmProjects/image_processing/lab1_python/rap.mp4"
video_out = "/Users/stone/PycharmProjects/image_processing/lab1_python/rap_out.avi"
# DoViolaJonesVideo(video, cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn, fn_out=video_out)

DoViolaJonesCamera(cascade_face_fn=cascade_face_fn, cascade_eye_fn=cascade_eye_fn, cascade_smile_fn=cascade_smile_fn)

