import cv2
import os
import matplotlib.pyplot as plt
import mediapipe as mp
import csv

# Path configurations
input_folder = "C:/Gait/Project2/test frames/left side walk"  # Folder with frames
output_folder = ""  # Folder to save annotated frames
csv_output_path = ""  # CSV file to save coordinates

# Create output directory if not exists
os.makedirs(output_folder, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Variables to store joint movements
all_joint_points = []

# List of joints to track
selected_joints = [11, 13, 15, 23, 25, 27]

# Get sorted list of frame files in the input folder
frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

# Open CSV file for writing
with open(csv_output_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header row
    header = ["Frame"]
    for joint_index in selected_joints:
        header.extend([f"Joint {joint_index} X", f"Joint {joint_index} Y"])
    csv_writer.writerow(header)

    for frame_count, frame_file in enumerate(frame_files):
        # Read the frame
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Unable to read {frame_path}. Skipping.")
            continue

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            joint_points = []
            row = [frame_count]
            for joint_index in selected_joints:
                landmark = results.pose_landmarks.landmark[joint_index]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                joint_points.append((x, y))
                row.extend([x, y])
                # Draw selected joints on the frame
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            all_joint_points.append(joint_points)
            # Write joint coordinates to CSV
            csv_writer.writerow(row)

        # Save the annotated frame
        output_frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_frame_path, frame)

pose.close()

# Plot joint movements
plt.figure(figsize=(10, 10))
for joint_index in range(len(selected_joints)):
    x_coords = [points[joint_index][0] for points in all_joint_points]
    y_coords = [points[joint_index][1] for points in all_joint_points]
    plt.scatter(x_coords, y_coords, label=f"Joint {selected_joints[joint_index]}")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Selected Joint Movements in XY Plane")
plt.legend()
plt.savefig("joint_movements_selected.png")
plt.show()