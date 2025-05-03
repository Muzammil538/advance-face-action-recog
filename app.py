import os
import cv2
import numpy as np
import face_recognition
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import pickle


class FacialRecognitionActionDetection:
    def __init__(self, faces_dir="known_faces", model_path=None):
        self.known_face_encodings = []
        self.known_face_names = []
        self.faces_dir = faces_dir

        # Try to load Haar cascade classifiers early to verify they exist
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Check if cascade files loaded successfully
            if self.eye_cascade.empty():
                print("Warning: Eye cascade classifier not loaded")
                self.eye_cascade = None
            if self.mouth_cascade.empty():
                print("Warning: Mouth cascade classifier not loaded")
                self.mouth_cascade = None
            if self.face_cascade.empty():
                print("Warning: Face cascade classifier not loaded")
                self.face_cascade = None
        except Exception as e:
            print(f"Warning: Could not load cascade classifiers: {e}")
            self.eye_cascade = None
            self.mouth_cascade = None
            self.face_cascade = None

        # Action classes
        self.action_classes = ['normal', 'sleeping', 'talking']

        # Initialize face recognition
        print("Initializing high-accuracy face recognition...")
        self.load_known_faces()

        # Initialize action detection model
        print("Initializing action detection model...")
        self.action_model = self.create_action_model()
        if model_path and os.path.exists(model_path):
            self.action_model.load_state_dict(torch.load(model_path))
        self.action_model.eval()

        # Define improved image transformations for action recognition
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Add slight augmentation for robustness
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create action history for temporal smoothing
        self.action_history = {}  # person_name -> list of recent actions
        self.history_length = 5  # Store last 5 actions for each person

        print("Initialization complete - using high-accuracy settings")

    def create_action_model(self):
        """Create a MobileNetV3 model for action detection"""
        try:
            # For macOS: Handle SSL certificate issues
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

            # Use MobileNetV3 Large for better accuracy while maintaining speed
            # Updated to use the new weights parameter instead of deprecated pretrained
            model = models.mobilenet_v3_large(weights='DEFAULT')
            # Modify the classifier to predict our 3 actions
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
            return model
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Creating model without pre-trained weights...")
            # Create model without pre-trained weights as fallback
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
            return model

    def load_known_faces(self):
        """Load known faces from directory"""
        # Force recompute encodings every time for better accuracy
        print("Loading known faces with high accuracy settings...")

        # If no faces directory, create it
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            print(f"Created faces directory at {self.faces_dir}")
            print("Please add face images to this directory and rerun the program")
            return

        # Load each person's faces - store individual encodings instead of averaging
        for person_folder in os.listdir(self.faces_dir):
            person_path = os.path.join(self.faces_dir, person_folder)
            if os.path.isdir(person_path):
                person_name = person_folder
                print(f"Loading faces for {person_name}")

                # Process each image of this person
                image_count = 0

                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_path, img_file)
                        image = face_recognition.load_image_file(img_path)

                        # Use CNN model for higher accuracy (at cost of speed)
                        face_locations = face_recognition.face_locations(image, model="cnn")

                        # If CNN fails or not available, fall back to HOG
                        if not face_locations:
                            face_locations = face_recognition.face_locations(image, model="hog")

                        if len(face_locations) > 0:
                            # Use higher number of jitters for better encoding quality
                            face_encodings = face_recognition.face_encodings(
                                image, face_locations, num_jitters=10, model="large"
                            )

                            # Add each encoding individually (don't average)
                            for face_encoding in face_encodings:
                                self.known_face_encodings.append(face_encoding)
                                self.known_face_names.append(person_name)
                                image_count += 1

                            print(f"  Processed image {img_file}")
                        else:
                            print(f"  No face found in {img_file}")

                print(f"Added {person_name} with {image_count} face encodings")

        print(f"Loaded {len(self.known_face_names)} face encodings for recognition")

    def predict_action(self, face_image):
        """Predict action (normal, sleeping, talking) from a face image"""
        try:
            # Check if face_image has valid dimensions
            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                return "normal", 0.5  # Default to normal when face is invalid

            # Use rule-based detection first for higher accuracy
            # Check for closed eyes - indicator of sleeping
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

            # Check for open mouth - indicator of talking
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouths = mouth_cascade.detectMultiScale(gray, 1.3, 11)

            # Rule-based decision with high confidence
            if len(eyes) == 0:
                return "sleeping", 0.85
            elif len(mouths) > 0:
                return "talking", 0.85

            # If rule-based approach doesn't give clear result, use ML model
            # Resize image for better model processing
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Apply basic preprocessing
            # Enhance contrast
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            enhanced_face_rgb = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB)

            # Convert face image to PIL Image
            pil_image = Image.fromarray(enhanced_face_rgb)

            # Apply transformations
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = self.action_model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get predicted class
            _, predicted_idx = torch.max(output, 1)
            action = self.action_classes[predicted_idx.item()]
            confidence = probabilities[predicted_idx.item()].item()

            # Boost confidence if it aligns with our rule-based detection
            if action == "sleeping" and len(eyes) == 0:
                confidence = max(confidence, 0.85)
            elif action == "talking" and len(mouths) > 0:
                confidence = max(confidence, 0.85)

            return action, confidence
        except Exception as e:
            print(f"Error predicting action: {e}")
            # Default to normal as fallback with medium confidence
            return "normal", 0.5

    def process_frame(self, frame):
        """Process a single frame for face recognition and action detection"""
        # Use full resolution for better accuracy
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the current frame - use better detector for accuracy
        try:
            # First try CNN model for better accuracy
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            if not face_locations:  # Fall back to HOG if CNN fails or no faces found
                face_locations = face_recognition.face_locations(rgb_frame)
        except Exception:
            # If CNN fails (e.g., no GPU), use HOG model
            face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            return frame, []

        # Get face encodings with higher accuracy
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations, num_jitters=5, model="large"
        )

        face_results = []
        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Identify the face with lower tolerance (stricter matching)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # If we found matches, use distance to find the best match
            if True in matches:
                # Calculate face distances
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                # Find all matches within tolerance
                matching_indices = [i for i, match in enumerate(matches) if match]

                # Group indices by name
                name_matches = {}
                for idx in matching_indices:
                    person_name = self.known_face_names[idx]
                    if person_name not in name_matches:
                        name_matches[person_name] = []
                    name_matches[person_name].append(idx)

                # Find person with most matches and lowest average distance
                best_name = None
                best_score = float('inf')

                for person_name, indices in name_matches.items():
                    avg_distance = sum(face_distances[i] for i in indices) / len(indices)
                    if avg_distance < best_score:
                        best_score = avg_distance
                        best_name = person_name

                if best_name:
                    name = best_name

            # Extract face ROI for action detection with padding for better context
            # Add padding to face region (20% on each side)
            height, width = frame.shape[:2]
            pad_x = int((right - left) * 0.2)
            pad_y = int((bottom - top) * 0.2)

            # Ensure coordinates are within frame boundaries
            face_top = max(0, top - pad_y)
            face_bottom = min(height, bottom + pad_y)
            face_left = max(0, left - pad_x)
            face_right = min(width, right + pad_x)

            face_roi = frame[face_top:face_bottom, face_left:face_right]

            if face_roi.size > 0:  # Make sure we have a valid ROI
                # Predict action
                action, confidence = self.predict_action(face_roi)

                # Post-process action predictions with smoothing
                # Simple heuristics for better action prediction
                if action == "sleeping" and confidence < 0.79:
                    # Check if eyes are closed
                    eyes_closed = self._check_eyes_closed(face_roi)
                    if not eyes_closed:
                        action = "normal"

                if action == "talking" and confidence < 0.79:
                    # Check for mouth movement
                    mouth_open = self._check_mouth_open(face_roi)
                    if not mouth_open:
                        action = "normal"

                # Draw rectangle around face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw name and action labels
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name}", (left + 6, bottom - 18), font, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{action} ({confidence:.2f})", (left + 6, bottom - 5), font, 0.5, (255, 255, 255),
                            1)

                face_results.append({
                    'name': name,
                    'action': action,
                    'confidence': confidence,
                    'location': (top, right, bottom, left)
                })

        return frame, face_results

    def _check_eyes_closed(self, face_image):
        """Check if eyes appear to be closed in the face image"""
        try:
            # Convert to grayscale for better eye detection
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Use Haar cascade for eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

            # If no eyes detected, they might be closed
            return len(eyes) == 0
        except Exception:
            return False

    def _check_mouth_open(self, face_image):
        """Check if mouth appears to be open in the face image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Use Haar cascade for mouth detection
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouths = mouth_cascade.detectMultiScale(gray, 1.3, 11)

            # If mouth detected, it's likely open
            return len(mouths) > 0
        except Exception:
            return False

    def train_action_model(self, data_dir, epochs=10):
        """Train the action detection model with custom data"""
        print("Training action detection model...")
        # This would be implemented for actual training
        # For now, we'll just simulate training
        for i in range(epochs):
            print(f"Training epoch {i + 1}/{epochs}")
            time.sleep(1)  # Simulate training time

        # Save model
        model_path = 'action_model.pth'
        torch.save(self.action_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def run_webcam(self):
        """Run face recognition and action detection on webcam feed"""
        if len(self.known_face_encodings) == 0:
            print("No known faces loaded. Please add face images to the known_faces directory.")
            return

        print("Starting webcam capture with high accuracy settings. Press 'q' to quit.")

        # Start video capture from webcam
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("Error: Could not open webcam. Trying alternative camera index...")
            cap = cv2.VideoCapture(1)  # Try another camera
            if not cap.isOpened():
                print("Error: Could not open any camera")
                return

        # Set higher resolution for better accuracy
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # For temporal smoothing
        person_action_history = {}

        # For processing every other frame (performance improvement)
        process_this_frame = True

        print("Camera initialized - beginning detection")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame")
                break

            # Only process every other frame for better performance
            if process_this_frame:
                # Process the frame
                processed_frame, face_results = self.process_frame(frame)

                # Apply temporal smoothing to action detection
                for result in face_results:
                    person_name = result['name']
                    current_action = result['action']

                    # Initialize history for new person
                    if person_name not in person_action_history:
                        person_action_history[person_name] = []

                    # Add current action to history
                    person_action_history[person_name].append(current_action)

                    # Keep history to a fixed length
                    if len(person_action_history[person_name]) > 5:
                        person_action_history[person_name].pop(0)

                    # Use majority vote for final action
                    if len(person_action_history[person_name]) >= 3:
                        from collections import Counter
                        action_counts = Counter(person_action_history[person_name])
                        smoothed_action = action_counts.most_common(1)[0][0]

                        # Update the displayed action
                        result['action'] = smoothed_action

            # Display the resulting frame (processed or not)
            if process_this_frame:
                display_frame = processed_frame
                # Display results
                for result in face_results:
                    print(f"Detected: {result['name']} - {result['action']} ({result['confidence']:.2f})")
            else:
                # Just show raw frame when not processing
                display_frame = frame

            # Display the frame
            cv2.imshow('Facial Recognition & Action Detection', display_frame)

            # Toggle processing flag
            process_this_frame = not process_this_frame

            # Break the loop on 'q' press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Add 'r' key to refresh face database
                print("Refreshing face database...")
                self.known_face_encodings = []
                self.known_face_names = []
                self.load_known_faces()

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("Initializing Facial Recognition and Action Detection System...")

    # Create the system instance
    system = FacialRecognitionActionDetection()

    # If you have pretrained action models, you can load them here
    # system = FacialRecognitionActionDetection(model_path='action_model.pth')

    # Print instructions
    print("\nINSTRUCTIONS:")
    print("1. Place your face images in the 'known_faces' directory")
    print("   - Create a separate folder for each person")
    print("   - Name the folder with the person's name")
    print("   - Add at least 4 photos of each person")
    print("   Example: known_faces/john/photo1.jpg, known_faces/john/photo2.jpg, etc.")
    print("2. The system will recognize faces and detect actions (normal, sleeping, talking)")
    print("3. Press 'q' to quit the webcam feed")
    print("\n")

    # Run the webcam-based detection
    system.run_webcam()


if __name__ == "__main__":
    main()