import cv2
import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected, assign an ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by removing IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with removed unused IDs
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# Load the Haar cascade XML file for player detection
player_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Open the video file
cap = cv2.VideoCapture("soccer_game.mov")

# Create the object tracker
tracker = EuclideanDistTracker()

while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper color thresholds for the desired color range
    lower_color = (45, 32, 72)
    upper_color = (189, 205, 246)

# Create a binary mask by applying the color range thresholds to the HSV frame
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Detect players using the Haar cascade classifier
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the bounding boxes of the detected players
    bounding_boxes = []
    for (x, y, w, h) in players:
        # Filter players based on color mask
        if cv2.countNonZero(color_mask[y:y+h, x:x+w]) > 0:
            bounding_boxes.append([x, y, w, h])

    # Perform object tracking
    boxes_ids = tracker.update(bounding_boxes)

    # Draw bounding boxes and IDs on the frame
    for box_id in boxes_ids:
        x, y, w, h, object_id = box_id
        cv2.putText(frame, str(object_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Check for the 'Esc' key to exit
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
