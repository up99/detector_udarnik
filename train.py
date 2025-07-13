from ultralytics import YOLO
def train_model():
    # Load a pretrained YOLOv11 model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Train the modelx
    results = model.train(data="data.yaml", epochs=120, device='cuda', degrees=180.0, patience=50)
    return results

if __name__ == '__main__':
    # Required for Windows when using PyTorch DataLoader with multiprocessing
    from multiprocessing import freeze_support
    freeze_support()

    train_model()