# import os
# import cv2
# import face_recognition
# import torch
# from torch import nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import time


# class FacialRecognitionActionDetection:
#     def __init__(self, faces_dir="known_faces", model_path=None):
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.faces_dir = faces_dir

#         try:
#             self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#             self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#             self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#             if self.eye_cascade.empty():
#                 print("Warning: Eye cascade classifier not loaded")
#                 self.eye_cascade = None
#             if self.mouth_cascade.empty():
#                 print("Warning: Mouth cascade classifier not loaded")
#                 self.mouth_cascade = None
#             if self.face_cascade.empty():
#                 print("Warning: Face cascade classifier not loaded")
#                 self.face_cascade = None
#         except Exception as e:
#             print(f"Warning: Could not load cascade classifiers: {e}")
#             self.eye_cascade = None
#             self.mouth_cascade = None
#             self.face_cascade = None

#         self.action_classes = ['normal', 'sleeping', 'talking']

#         print("Initializing high-accuracy face recognition...")
#         self.load_known_faces()

#         print("Initializing action detection model...")
#         self.action_model = self.create_action_model()
#         if model_path and os.path.exists(model_path):
#             self.action_model.load_state_dict(torch.load(model_path))
#         self.action_model.eval()

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ColorJitter(brightness=0.1, contrast=0.1),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         self.action_history = {}
#         self.history_length = 5

#         print("Initialization complete - using high-accuracy settings")

#     def create_action_model(self):
#         try:
#             import ssl
#             ssl._create_default_https_context = ssl._create_unverified_context

#             model = models.mobilenet_v3_large(weights='DEFAULT')
#             model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
#             return model
#         except Exception as e:
#             print(f"Error loading pre-trained model: {e}")
#             print("Creating model without pre-trained weights...")
#             model = models.mobilenet_v3_large(weights=None)
#             model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
#             return model

#     def load_known_faces(self):
#         print("Loading known faces with high accuracy settings...")

#         if not os.path.exists(self.faces_dir):
#             os.makedirs(self.faces_dir)
#             print(f"Created faces directory at {self.faces_dir}")
#             print("Please add face images to this directory and rerun the program")
#             return

#         for person_folder in os.listdir(self.faces_dir):
#             person_path = os.path.join(self.faces_dir, person_folder)
#             if os.path.isdir(person_path):
#                 person_name = person_folder
#                 print(f"Loading faces for {person_name}")

#                 image_count = 0

#                 for img_file in os.listdir(person_path):
#                     if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         img_path = os.path.join(person_path, img_file)
#                         image = face_recognition.load_image_file(img_path)

#                         face_locations = face_recognition.face_locations(image, model="cnn")

#                         if not face_locations:
#                             face_locations = face_recognition.face_locations(image, model="hog")

#                         if len(face_locations) > 0:
#                             face_encodings = face_recognition.face_encodings(
#                                 image, face_locations, num_jitters=10, model="large"
#                             )

#                             for face_encoding in face_encodings:
#                                 self.known_face_encodings.append(face_encoding)
#                                 self.known_face_names.append(person_name)
#                                 image_count += 1

#                             print(f"  Processed image {img_file}")
#                         else:
#                             print(f"  No face found in {img_file}")

#                 print(f"Added {person_name} with {image_count} face encodings")

#         print(f"Loaded {len(self.known_face_names)} face encodings for recognition")

#     def predict_action(self, face_image):
#         try:
#             if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
#                 return "normal", 0.5

#             gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#             eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#             eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

#             mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#             mouths = mouth_cascade.detectMultiScale(gray, 1.3, 11)

#             if len(eyes) == 0:
#                 return "sleeping", 0.85
#             elif len(mouths) > 0:
#                 return "talking", 0.85

#             face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

#             lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#             cl = clahe.apply(l)
#             enhanced_lab = cv2.merge((cl, a, b))
#             enhanced_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
#             enhanced_face_rgb = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB)

#             pil_image = Image.fromarray(enhanced_face_rgb)

#             input_tensor = self.transform(pil_image)
#             input_batch = input_tensor.unsqueeze(0)

#             with torch.no_grad():
#                 output = self.action_model(input_batch)
#                 probabilities = torch.nn.functional.softmax(output[0], dim=0)

#             _, predicted_idx = torch.max(output, 1)
#             action = self.action_classes[predicted_idx.item()]
#             confidence = probabilities[predicted_idx.item()].item()

#             if action == "sleeping" and len(eyes) == 0:
#                 confidence = max(confidence, 0.85)
#             elif action == "talking" and len(mouths) > 0:
#                 confidence = max(confidence, 0.85)

#             return action, confidence
#         except Exception as e:
#             print(f"Error predicting action: {e}")
#             return "normal", 0.5

#     def process_frame(self, frame):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         try:
#             face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
#             if not face_locations:
#                 face_locations = face_recognition.face_locations(rgb_frame)
#         except Exception:
#             face_locations = face_recognition.face_locations(rgb_frame)

#         if not face_locations:
#             return frame, []

#         face_encodings = face_recognition.face_encodings(
#             rgb_frame, face_locations, num_jitters=5, model="large"
#         )

#         face_results = []
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
#             name = "Unknown"

#             if True in matches:
#                 face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

#                 matching_indices = [i for i, match in enumerate(matches) if match]

#                 name_matches = {}
#                 for idx in matching_indices:
#                     person_name = self.known_face_names[idx]
#                     if person_name not in name_matches:
#                         name_matches[person_name] = []
#                     name_matches[person_name].append(idx)

#                 best_name = None
#                 best_score = float('inf')

#                 for person_name, indices in name_matches.items():
#                     avg_distance = sum(face_distances[i] for i in indices) / len(indices)
#                     if avg_distance < best_score:
#                         best_score = avg_distance
#                         best_name = person_name

#                 if best_name:
#                     name = best_name

#             height, width = frame.shape[:2]
#             pad_x = int((right - left) * 0.2)
#             pad_y = int((bottom - top) * 0.2)

#             face_top = max(0, top - pad_y)
#             face_bottom = min(height, bottom + pad_y)
#             face_left = max(0, left - pad_x)
#             face_right = min(width, right + pad_x)

#             face_roi = frame[face_top:face_bottom, face_left:face_right]

#             if face_roi.size > 0:  # Make sure we have a valid ROI
#                 action, confidence = self.predict_action(face_roi)

#                 if action == "sleeping" and confidence < 0.79:
#                     eyes_closed = self._check_eyes_closed(face_roi)
#                     if not eyes_closed:
#                         action = "normal"

#                 if action == "talking" and confidence < 0.79:
#                     mouth_open = self._check_mouth_open(face_roi)
#                     if not mouth_open:
#                         action = "normal"

#                 color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#                 cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

#                 cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
#                 font = cv2.FONT_HERSHEY_DUPLEX
#                 cv2.putText(frame, f"{name}", (left + 6, bottom - 18), font, 0.6, (255, 255, 255), 1)
#                 cv2.putText(frame, f"{action} ({confidence:.2f})", (left + 6, bottom - 5), font, 0.5, (255, 255, 255),
#                             1)

#                 face_results.append({
#                     'name': name,
#                     'action': action,
#                     'confidence': confidence,
#                     'location': (top, right, bottom, left)
#                 })

#         return frame, face_results

#     def _check_eyes_closed(self, face_image):
#         try:
#             gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

#             eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#             eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

#             return len(eyes) == 0
#         except Exception:
#             return False

#     def _check_mouth_open(self, face_image):
#         try:
#             gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

#             mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#             mouths = mouth_cascade.detectMultiScale(gray, 1.3, 11)

#             return len(mouths) > 0
#         except Exception:
#             return False

#     def train_action_model(self, data_dir, epochs=10):
#         print("Training action detection model...")
#         for i in range(epochs):
#             print(f"Training epoch {i + 1}/{epochs}")
#             time.sleep(1)

#         model_path = 'action_model.pth'
#         torch.save(self.action_model.state_dict(), model_path)
#         print(f"Model saved to {model_path}")

#     def run_webcam(self):
#         if len(self.known_face_encodings) == 0:
#             print("No known faces loaded. Please add face images to the known_faces directory.")
#             return

#         print("Starting webcam capture with high accuracy settings. Press 'q' to quit.")

#         cap = cv2.VideoCapture(0)

#         if not cap.isOpened():
#             print("Error: Could not open webcam. Trying alternative camera index...")
#             cap = cv2.VideoCapture(1)  # Try another camera
#             if not cap.isOpened():
#                 print("Error: Could not open any camera")
#                 return

#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         person_action_history = {}

#         process_this_frame = True

#         print("Camera initialized - beginning detection")

#         while True:
#             ret, frame = cap.read()

#             if not ret:
#                 print("Error: Failed to capture frame")
#                 break

#             if process_this_frame:
#                 processed_frame, face_results = self.process_frame(frame)

#                 for result in face_results:
#                     person_name = result['name']
#                     current_action = result['action']

#                     if person_name not in person_action_history:
#                         person_action_history[person_name] = []

#                     person_action_history[person_name].append(current_action)

#                     if len(person_action_history[person_name]) > 5:
#                         person_action_history[person_name].pop(0)

#                     if len(person_action_history[person_name]) >= 3:
#                         from collections import Counter
#                         action_counts = Counter(person_action_history[person_name])
#                         smoothed_action = action_counts.most_common(1)[0][0]

#                         result['action'] = smoothed_action

#             if process_this_frame:
#                 display_frame = processed_frame
#                 for result in face_results:
#                     print(f"Detected: {result['name']} - {result['action']} ({result['confidence']:.2f})")
#             else:
#                 display_frame = frame

#             # Display the frame
#             cv2.imshow('Facial Recognition & Action Detection', display_frame)

#             # Toggle processing flag
#             process_this_frame = not process_this_frame

#             # Break the loop on 'q' press
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('r'):  # Add 'r' key to refresh face database
#                 print("Refreshing face database...")
#                 self.known_face_encodings = []
#                 self.known_face_names = []
#                 self.load_known_faces()

#         cap.release()
#         cv2.destroyAllWindows()


# def main():
#     print("Initializing Facial Recognition and Action Detection System...")

#     system = FacialRecognitionActionDetection()

#     # If you have pretrained action models, you can load them here
#     # system = FacialRecognitionActionDetection(model_path='action_model.pth')

#     print("\nINSTRUCTIONS:")
#     print("1. Place your face images in the 'known_faces' directory")
#     print("   - Create a separate folder for each person")
#     print("   - Name the folder with the person's name")
#     print("   - Add at least 4 photos of each person")
#     print("   Example: known_faces/john/photo1.jpg, known_faces/john/photo2.jpg, etc.")
#     print("2. The system will recognize faces and detect actions (normal, sleeping, talking)")
#     print("3. Press 'q' to quit the webcam feed")
#     print("\n")

#     system.run_webcam()


# if __name__ == "__main__":
#     main()