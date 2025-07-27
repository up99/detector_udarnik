import cv2
import os
import shutil
from torchvision import transforms as T
from PIL import Image
import numpy as np

def augment(dataset_path, subsets=['train', 'val', 'test']):
    """
    Augment YOLO dataset by inverting pixel values (bitwise NOT) and T.Normalize
    
    Args:
        dataset_path (str): Path to the dataset root directory
        subsets (list): List of subset folders to process (train, val, test)
    """
    
    for subset in subsets:
        images_path = os.path.join(dataset_path, subset, 'images')
        labels_path = os.path.join(dataset_path, subset, 'labels')
        
        # Check if directories exist
        if not os.path.exists(images_path):
            print(f"Warning: {images_path} does not exist, skipping...")
            continue
            
        if not os.path.exists(labels_path):
            print(f"Warning: {labels_path} does not exist, skipping...")
            continue
        
        print(f"Processing {subset} set...")
        
        # Process all images in the images directory
        for filename in os.listdir(images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get base filename without extension
                base_name = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[1]
                
                # Full paths
                img_path = os.path.join(images_path, filename)
                
                # Check if corresponding label file exists
                label_filename = base_name + '.txt'
                label_path = os.path.join(labels_path, label_filename)
                
                if not os.path.exists(label_path):
                    print(f"Warning: Label file {label_filename} not found for {filename}, skipping...")
                    continue
                
                # Read image
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"Warning: Could not read image {filename}, skipping...")
                    continue
                image_T =  T.ToTensor()(Image.open(img_path).convert("RGB"))

                # Apply bitwise NOT augmentation and T.notmalization
                img_inverted = cv2.bitwise_not(img_bgr)
                img_norm = (T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_T).unsqueeze(0)).squeeze(0).numpy().transpose(1, 2, 0)
                img_cv2_norm = (np.round(cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX)))[:, :, ::-1]
                
                img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                # equalize the histogram of the Y channel
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                # convert the YUV image back to RGB format
                img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                # Save augmented image with _bitwise suffix
                augmented_img_name = f"{base_name}_bitwise{extension}"
                augmented_img_path = os.path.join(images_path, augmented_img_name)
                cv2.imwrite(augmented_img_path, img_inverted)
                
                augmented_imgT_name = f"{base_name}_normalized{extension}"
                augmented_imgT_path = os.path.join(images_path, augmented_imgT_name)
                cv2.imwrite(augmented_imgT_path, img_cv2_norm)

                augmented_img_eq_name = f"{base_name}_equalized{extension}"
                augmented_img_eq_path = os.path.join(images_path, augmented_img_eq_name)
                cv2.imwrite(augmented_img_eq_path, img_equalized)

                # Copy label file with _bitwise suffix (labels remain the same)
                augmented_label_name = f"{base_name}_bitwise.txt"
                augmented_label_path = os.path.join(labels_path, augmented_label_name)
                shutil.copy2(label_path, augmented_label_path)
                
                augmented_labelT_name = f"{base_name}_normalized.txt"
                augmented_labelT_path = os.path.join(labels_path, augmented_labelT_name)
                shutil.copy2(label_path, augmented_labelT_path)

                augmented_label_eq_name = f"{base_name}_equalized.txt"
                augmented_label_eq_path = os.path.join(labels_path, augmented_label_eq_name)
                shutil.copy2(label_path, augmented_label_eq_path)

                print(f"Augmented: {filename} -> {augmented_img_name}")

def main():
    dataset_path = "dataset_augmented/" 
    subsets_to_process = ['train', 'val', 'test']
    print("Starting YOLO dataset augmentation...")
    augment(dataset_path, subsets_to_process)
    print("Augmentation completed!")

if __name__ == "__main__":
    main()