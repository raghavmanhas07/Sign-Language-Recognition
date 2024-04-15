from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

# Load the trained model for Marathi sign language gestures
model_dict = pickle.load(open('./model_marathi.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define label dictionary for Marathi characters
labels_dict = {
    0: 'अ', 1: 'आ', 2: 'इ', 3: 'ई', 4: 'उ', 5: 'ऊ', 6: 'ए', 7: 'ऐ', 
    8: 'ओ', 9: 'औ', 10: 'क', 11: 'ख', 12: 'ग', 13: 'घ', 14: 'च', 15: 'छ', 
    16: 'ज', 17: 'झ', 18: 'ञ', 19: 'ट', 20: 'ठ', 21: 'ड', 22: 'ढ', 23: 'ण', 
    24: 'त', 25: 'थ', 26: 'द', 27: 'ध', 28: 'न', 29: 'प', 30: 'फ', 31: 'ब', 
    32: 'भ', 33: 'म', 34: 'य', 35: 'र', 36: 'ल', 37: 'व', 38: 'श', 39: 'ष', 
    40: 'स', 41: 'ह', 42: 'ळ'
}

# Specify the path to the Marathi font file
marathi_font_path = r'C:\Users\HP\Downloads\sign-language-detector-python-master (1)/Mangal Regular.ttf'

try:
    # Load the Marathi font
    marathi_font = ImageFont.truetype(marathi_font_path, 48)  # Adjust the font size as needed
    print("Marathi font loaded successfully:")
    print("Font file:", marathi_font_path)
    print("Font properties:", marathi_font.getname())
except Exception as e:
    print("Error loading Marathi font:", e)
    exit()

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read frame from the video capture
    ret, frame = cap.read()
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make prediction using the trained model
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            print("Predicted character:", predicted_character)
        except Exception as e:
            print("Error predicting character:", e)
            predicted_character = ''

        # Use PIL to render Marathi character
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Adjust text position (centered horizontally, slightly above the bounding box)
        text_position = (x1, y1 - 10)
        if text_position[0] < 0 or text_position[1] < 0 or text_position[0] >= W or text_position[1] >= H:
             print("Text position is outside the frame bounds.")
        else:
            print("Text position is within the frame bounds.")
        # Render text on the PIL image
        draw.text(text_position, predicted_character, font=marathi_font, fill=(0, 0, 0))  # Use white color

        # Convert back to OpenCV format and display the frame
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('frame', frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
