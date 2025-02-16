import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('/Users/hanafahim/Desktop/repos/mrm/cnn')
from model import MNISTModel


# Load trained CNN model
model = MNISTModel(1, 10, 10)
model.load_state_dict(torch.load("/Users/hanafahim/Desktop/repos/mrm/cnn/weights/mnist_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Load HSV values if available
loadFromSys = True
if loadFromSys:
    hsv_value = np.load('hsv_value.npy')

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)  
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Persistent canvas

x1, y1 = 0, 0
noise_thresh = 800

# Define HSV range (manual or loaded)
lower_range = np.array([105, 150, 150])
upper_range = np.array([179, 255, 255])

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 30)  # Retain strokes

        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0  # Reset tracking

    # Overlay canvas on the frame
    frame_with_canvas = cv2.add(frame, canvas)

    # Display the images
    stacked = np.hstack((canvas, frame_with_canvas))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))
    cv2.imshow("Mask", mask)

    # Handle keypress events
    key = cv2.waitKey(1)
    
    if key == 10:  # Exit on 'Enter'
        break
    
    elif key == ord('p'):  # Predict digit
         digit_img = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)  # Ensure grayscale
         digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)  # Resize directly

    # Show processed image
         cv2.imshow("Processed Handwritten Digit", digit_img)

    # Convert to tensor for model
         import torchvision.transforms as transforms

# Define transformation (convert to tensor and normalize)
         transform = transforms.Compose([
    transforms.ToTensor(),  # Converts (H, W) numpy array to (1, H, W) tensor and scales [0,255] → [0,1]
])

         digit_tensor = transform(digit_img).unsqueeze(0)
         with torch.no_grad():
              output = model(digit_tensor)
              predicted_digit = torch.argmax(F.softmax(output, dim=1)).item()

         print(f"Predicted Digit: {predicted_digit}")
    

    elif key == ord('c'):  # Clear canvas
        canvas.fill(0)

# Cleanup
cap.release()
cv2.destroyAllWindows()

