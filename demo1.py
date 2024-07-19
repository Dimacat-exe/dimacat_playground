import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from pyunpack import Archive
from ultralytics import YOLO

# Set up the environment
def setup_environment():
    import locale
    locale.getpreferredencoding = lambda: "UTF-8"
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    drive.mount('/content/drive')
    print("Environment set up successfully!")

# Extract and prepare the dataset
def prepare_dataset():
    Archive('/content/drive/MyDrive/images_labeled.rar').extractall('/content')
    print("Dataset extracted successfully!")
    return {
        'train_images': "/content/images_labeled/train/images",
        'train_labels': "/content/images_labeled/train/labels",
        'val_images': "/content/images_labeled/valid/images",
        'val_labels': "/content/images_labeled/valid/labels"
    }

# Display first 30 images with bounding boxes
def show_first_images(data_paths, num_images=30):
    image_files = sorted(os.listdir(data_paths['train_images']))[:num_images]
    
    fig, axs = plt.subplots(5, 6, figsize=(20, 16))
    for idx, img_file in enumerate(image_files):
        img = cv2.imread(os.path.join(data_paths['train_images'], img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label_file = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(data_paths['train_labels'], label_file), "r") as f:
            labels = f.read().strip().split("\n")
        
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.split())
            x_min, y_min = int((x_center - width/2) * img.shape[1]), int((y_center - height/2) * img.shape[0])
            x_max, y_max = int((x_center + width/2) * img.shape[1]), int((y_center + height/2) * img.shape[0])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        axs[idx//6, idx%6].imshow(img)
        axs[idx//6, idx%6].axis('off')
    
    # Turn off any unused subplots
    for i in range(idx+1, 30):
        axs[i//6, i%6].axis('off')
    
    plt.tight_layout()
    plt.show()

# Train the YOLO model
def train_yolo_model(epochs=25, img_size=640):
    model = YOLO("yolov8n.yaml")
    model.train(data='/content/images_labeled/data.yaml', epochs=epochs, imgsz=img_size)
    print(f"Model trained for {epochs} epochs!")
    return model

# Visualize training metrics
def visualize_training_metrics():
    results = pd.read_csv('/content/runs/detect/train/results.csv')
    metrics = ['box_loss', 'cls_loss', 'dfl_loss', 'precision', 'recall', 'mAP50', 'mAP50-95']
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    for idx, metric in enumerate(metrics):
        ax = axs[idx//2, idx%2]
        for prefix in ['train', 'val']:
            if f'{prefix}/{metric}' in results.columns:
                sns.lineplot(x='epoch', y=f'{prefix}/{metric}', data=results, ax=ax, label=prefix)
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Evaluate the model and show results
def evaluate_model(model):
    metrics = model.val()
    print("Model Evaluation Results:")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP75: {metrics.box.map75:.3f}")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['mAP50-95', 'mAP50', 'mAP75'], 
                y=[metrics.box.map, metrics.box.map50, metrics.box.map75])
    plt.title('YOLO Evaluation Metrics')
    plt.ylabel('Value')
    plt.show()

# Show confusion matrix
def show_confusion_matrix():
    matrix = plt.imread('/content/runs/detect/val/confusion_matrix.png')
    plt.figure(figsize=(12, 12))
    plt.imshow(matrix)
    plt.axis('off')
    plt.title('Confusion Matrix')
    plt.show()

# Detect cats in custom images
def detect_cats(model, image_folder, num_images=20):
    image_files = os.listdir(image_folder)
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    fig, axs = plt.subplots(5, 4, figsize=(20, 25))
    for idx, img_file in enumerate(selected_images):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        results = model(img)
        detected_img = results[0].plot()
        detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        
        axs[idx//4, idx%4].imshow(detected_img)
        axs[idx//4, idx%4].axis('off')
    
    # Turn off any unused subplots
    for i in range(len(selected_images), 20):
        axs[i//4, i%4].axis('off')
    
    plt.tight_layout()
    plt.show()

# Process video with cat detection
def process_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=0.1)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print("Video processing complete!")

# Main execution
if __name__ == "__main__":
    setup_environment()
    data_paths = prepare_dataset()
    
    print("Displaying first 30 images from the dataset...")
    show_first_images(data_paths)
    
    print("Training YOLO model...")
    model = train_yolo_model()
    
    print("Visualizing training metrics...")
    visualize_training_metrics()
    
    print("Evaluating the model...")
    evaluate_model(model)
    
    print("Displaying confusion matrix...")
    show_confusion_matrix()
    
    print("Detecting cats in custom images...")
    detect_cats(model, '/content/test')
    
    print("Processing video...")
    process_video(model, '/content/drive/MyDrive/input_video.mp4', 'output_video.mp4')
    
    print("Saving results...")
    os.system('zip -r "/content/results.zip" "/content/runs/"')
    os.system('cp "/content/results.zip" "/content/drive/MyDrive/"')
    os.system('cp "/content/output_video.mp4" "/content/drive/MyDrive/"')
    
    print("All tasks completed successfully!")