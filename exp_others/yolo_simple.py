import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO("yolov8s.pt")

# # Load the video capture
# videoCap = cv2.VideoCapture(0)


def yolo_track_and_draw_box(frame):
    results = yolo.track(frame, stream=True)

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(
                    frame,
                    f"{class_name} {box.conf[0]:.2f}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    colour,
                    2,
                )


# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (cls_num // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)


print("reading img")
image = cv2.imread("test3.jpg")

yolo_track_and_draw_box(image)

# show the image
cv2.imshow("frame", image)

print("done")
# break the loop if 'q' is pressed
cv2.waitKey(0) & 0xFF == ord("q")


# # release the video capture and destroy all windows
# videoCap.release()
# cv2.destroyAllWindows()
