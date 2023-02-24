import cv2
import numpy as np

class MonsterDetector:
    def __init__(self, screen_width, screen_height, monster_template_path):
        self.cap = cv2.VideoCapture(0)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.roi = (0, 0, screen_width, screen_height)
        self.monster_template = cv2.imread(monster_template_path, cv2.IMREAD_GRAYSCALE)

    def detect_monsters(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Failed to capture frame")
            return None

        # Crop the frame to the ROI
        x, y, w, h = self.roi
        frame_roi = frame[y:y+h, x:x+w]

        # Convert the ROI to grayscale
        frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        # Match the monster template to the grayscale ROI
        result = cv2.matchTemplate(frame_gray, self.monster_template, cv2.TM_CCOEFF_NORMED)

        # Threshold the result to get the locations of the monsters
        threshold = 0.8
        locations = np.where(result >= threshold)

        # Draw rectangles around the monsters
        for loc in zip(*locations[::-1]):
            cv2.rectangle(frame_roi, loc, (loc[0]+self.monster_template.shape[1], loc[1]+self.monster_template.shape[0]), (0, 255, 0), 2)

        # Display the ROI with rectangles around the monsters
        cv2.imshow("ROI", frame_roi)

        # Wait for keypress to exit the loop
        if cv2.waitKey(1) == ord('q'):
            return None

        # Return the locations of the monsters
        return locations
