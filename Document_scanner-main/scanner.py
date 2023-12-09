import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

# Set up video capture
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize variables
count = 0
scale = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX

WIDTH, HEIGHT = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# Function for basic image processing
def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold

# Function for detecting a document in the frame
def scan_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Add a check to ensure contours are found
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
                if area > max_area and len(approx) == 4:
                    document_contour = approx
                    max_area = area

        cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)


# Function to center text on the image
def center_text(image, text):
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, 2, (255, 0, 255), 5, cv2.LINE_AA)

# Main loop
while True:

    _, frame = cap.read()  # Capture a frame
    frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotate the frame 180 degrees
    frame_copy = frame.copy()  # Create a copy for processing

    # Detect the document in the frame
    scan_detection(frame_copy)

     # Display the input frame
    cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))

    # Perform perspective transformation on the document
    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    cv2.imshow("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))

    # Process the document image
    processed = image_processing(warped)
    processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
    cv2.imshow("Processed", cv2.resize(processed, (int(scale * processed.shape[1]),
                                                  int(scale * processed.shape[0]))))
    # Check for user input
    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == 27:  # 'Esc' key to exit
        break

    elif pressed_key == ord('s'):  # 's' key to save the scanned document
        cv2.imwrite("output/scanned_" + str(count) + ".jpg", processed)
        count += 1

        # Display a message on the input frame
        center_text(frame, "Scan Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv2.waitKey(500)  # Display the message for 500 milliseconds

    elif pressed_key == ord('o'):  # 'o' key to perform OCR and save text
        file = open("output/recognized_" + str(count - 1) + ".txt", "w")
        ocr_text = pytesseract.image_to_string(warped)
        # print(ocr_text)
        file.write(ocr_text)
        file.close()

        # Display a message on the input frame
        center_text(frame, "Text Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
        cv2.waitKey(800)  # Display the message for 500 milliseconds

# Clean up
cv2.destroyAllWindows()
