#HRNet_________________________________________________________________________________________
import cv2
import os
import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Load the HRNet model (Keypoint R-CNN)
model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing function for the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Detect keypoints from a frame using HRNet
def get_keypoints(frame, model):
    # Preprocess the image and pass it through the HRNet model
    input_tensor = preprocess_image(frame)
    with torch.no_grad():
        outputs = model(input_tensor)  # Get model predictions

    # Extract keypoints for the first person detected (if any)
    if len(outputs[0]['keypoints']) > 0:
        keypoints = outputs[0]['keypoints'][0].cpu().numpy()  # Keypoints for the first detected person
        # Select only the desired left-side joints (indices: 5, 7, 9, 11, 13, 15)
        selected_indices = [5, 7, 9, 11, 13, 15]
        selected_keypoints = keypoints[selected_indices]
        return selected_keypoints
    return None

# Save frame with keypoints plotted
def save_frame_with_keypoints(frame, keypoints, output_folder, frame_id, threshold=0.5):
    if keypoints is not None:
        for x, y, confidence in keypoints:
            if confidence > threshold:  # Confidence threshold
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoints
    output_path = os.path.join(output_folder, f'frame_{frame_id:04d}.png')
    cv2.imwrite(output_path, frame)

# Save keypoints to CSV file
def save_keypoints_to_csv(keypoints, frame_id, csv_writer, threshold=0.5):
    if keypoints is not None:
        for idx, (x, y, confidence) in enumerate(keypoints):
            if confidence > threshold:  # Confidence threshold
                csv_writer.writerow([frame_id, idx, x, y])  # frame_id, joint_id, x, y

# Plot joints on an XY plane
def plot_joints(joint_data):
    x = [joint[0] for joint in joint_data]
    y = [joint[1] for joint in joint_data]
    plt.scatter(x, y)
    plt.title('Joint Movement in XY Plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Main function to process frames
def process_frames(input_folder, output_folder, csv_file_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the CSV file to write keypoints data
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame_ID', 'Joint_ID', 'X_Coord', 'Y_Coord'])  # CSV header

        frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
        joint_data = []  # To store joint positions for XY plane scatter plot

        for frame_id, frame_file in enumerate(frame_files):
            frame_path = os.path.join(input_folder, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Warning: Unable to read {frame_path}. Skipping.")
                continue

            # Detect keypoints for the frame
            keypoints = get_keypoints(frame, model)

            # Save frame with keypoints
            save_frame_with_keypoints(frame, keypoints, output_folder, frame_id)

            # Save keypoints data to CSV
            save_keypoints_to_csv(keypoints, frame_id, csv_writer)

            # Collect joint data (using keypoints of joints)
            if keypoints is not None:
                for x, y, confidence in keypoints:  # Only add valid keypoints
                    joint_data.append([x, y])

        # Plot joint movement in XY plane
        plot_joints(joint_data)

    print(f"Keypoints saved to {csv_file_path}")

# Run the function with your folder of frames
input_folder = ""  # Replace with your frames folder path
output_folder = ""   # Folder to save annotated frames
csv_file_path = ""  # Path to save the CSV file
process_frames(input_folder, output_folder, csv_file_path)


