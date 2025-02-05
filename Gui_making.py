from tkinter import *
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
import os

current_dir = os.path.dirname(__file__)


# Global variables
window, web_cam_label, entry = None, None, None
camera, video_capture = None, None
current_mode = None

# We Load YOLO model
net = cv.dnn.readNet(os.path.join(current_dir, "yolov4-tiny_final.weights"),os.path.join(current_dir,  "yolov4-tiny.cfg"))
with open("obj.names", "r") as f:
    classes = f.read().strip().split("\n")  # Load class names

# bounding boxes color
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def close():
    global camera, video_capture
    print("You have closed the program")
    if camera is not None:
        camera.release()  # Release the camera if it's open
    if video_capture is not None:
        video_capture.release()  # Release the video if it's open
    window.destroy()

def detect_objects(frame):

    height, width = frame.shape[:2]

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Scale the bounding box coordinates to the original frame size
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                # Calculate top-left corner of the bounding box
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = COLORS[class_ids[i]]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = f"{label}: {confidence:.2f}"
            cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def update_video_stream():
    global web_cam_label, photo, camera, video_capture, current_mode

    if current_mode == "camera" and camera is not None:
        ret, frame = camera.read()
        if ret:
            # Perform object detection on the frame
            frame = detect_objects(frame)

            # Convert the OpenCV frame to PIL format
            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            # Resize the image to fit the label
            image = image.resize((600, 480))
            # Convert the PIL image to Tkinter format
            photo = ImageTk.PhotoImage(image)
            web_cam_label.config(image=photo)
            web_cam_label.image = photo

    elif current_mode == "video" and video_capture is not None:
        ret, frame = video_capture.read()
        if ret:
            frame = detect_objects(frame)
            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            # Resize the image to fit the label
            image = image.resize((600, 480))
            # Convert the PIL image to Tkinter format
            photo = ImageTk.PhotoImage(image)
            web_cam_label.config(image=photo)
            web_cam_label.image = photo
        else:
            video_capture.release()
            video_capture = None
            current_mode = None
            return

    window.after(1, update_video_stream)

def start_camera():
    global camera, video_capture, current_mode

    if video_capture is not None:
        video_capture.release()
        video_capture = None

    if camera is None:
        camera = cv.VideoCapture(0)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv.CAP_PROP_FPS, 30)  # Set frame rate to 30 FPS

    current_mode = "camera"
    update_video_stream()  # Start updating the video stream

def start_video():
    global camera, video_capture, current_mode, entry

    if camera is not None:
        camera.release()
        camera = None

    # Get the video path from the entry widget
    video_path = entry.get()

    # Start the video
    if video_capture is None:
        video_capture = cv.VideoCapture(video_path)  # Opens the video file

    current_mode = "video"
    update_video_stream()  # Start updating the video stream

def delete():
    entry.delete(0, END)
def show_tkinter_gui():
    global window, web_cam_label, entry

    window = Tk()
    window.geometry("960x580")
    window.title("Bondstein Project")

    entry = Entry(window, font=("Arial,10"))
    entry.place(x=50, y=231)

    del_button = Button(window, text="delete", command=delete)
    del_button.place(x=165, y=260)

    submit_button = Button(window, text="Submit", command=start_video)
    submit_button.place(x=80, y=260)

    verify_button = Button(window, text="Track on Camera", bg='#69154a', relief=RAISED, bd=5, command=start_camera)
    verify_button.place(x=85, y=320)

    close_button = Button(window, text="Close the Program", bg='#69155a', relief=RAISED, bd=5, command=close)
    close_button.place(x=80, y=400)

    # Welcome label with image
    image = Image.open(os.path.join(current_dir, "Bus.jpg"))
    image = image.resize((200, 100))
    photo2 = ImageTk.PhotoImage(image)
    label = Label(window, text="Welcome to Detection Management \n Application", font=('Arial', 10, 'bold'),
                  foreground="green", bg='#332e23',
                  relief=RAISED,
                  bd=5,
                  padx=5,
                  pady=5, image=photo2, compound='bottom')
    label.place(x=9, y=9)

    label1 = Label(window, text="Enter the video path to detect object \n e.g. path/to/video ", font=('Arial', 10, 'bold'),
                   foreground="green", bg='#c4a154')
    label1.place(x=20, y=180)

    web_cam_label = Label(window, text="webcam/video")
    web_cam_label.place(x=300, y=20, width=630, height=500)

    window.protocol("WM_DELETE_WINDOW", close)

    window.iconphoto(True, photo2)
    window.config(background="#71857d")

    window.mainloop()

# Run the GUI
show_tkinter_gui()