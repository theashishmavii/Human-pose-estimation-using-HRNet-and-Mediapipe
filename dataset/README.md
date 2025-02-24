# 📂 Human Pose Estimation Dataset  

## 📌 Overview  

This folder is **intended** to contain datasets for **training human pose estimation models**. However, in this project, **you don’t need to train the model manually**.  

Instead, you can **directly use the provided scripts** on **any image or video** to get the results without requiring additional datasets.  

---

## 📌 Why is the Dataset Not Provided?  

✅ **Pretrained Models** – The project utilizes **HRNet & MediaPipe**, which already have trained models.  
✅ **No Training Required** – The provided scripts allow you to **run pose estimation on any custom images/videos**.  
✅ **Flexibility** – Instead of a fixed dataset, you can **test the model on your own media files**.  

---

## 📌 How to Use Your Own Dataset?  

1️⃣ **Prepare Images or Videos**  
   - Ensure your media files are in standard formats (e.g., `.jpg`, `.png`, `.mp4`).  

2️⃣ **Run the Provided Scripts**  
   - **For HRNet-based Pose Estimation**  
     ```sh
     python scripts/hrnet_main_code.py --input your_video.mp4
     ```
   - **For MediaPipe-based Pose Estimation**  
     ```sh
     python scripts/mediapipe_main_code.py --input your_image.jpg
     ```

3️⃣ **View the Results**  
   - The model will **detect keypoints** and display the **visualized pose estimation**.  

---

## 📌 Want to Train on a Custom Dataset?  

If you still want to **train the model from scratch**, you can:  
🔹 Use publicly available datasets like **COCO Keypoints**, **MPII**, or **PoseTrack**.  
🔹 Modify the scripts to load custom **annotated datasets** for fine-tuning.  

---

