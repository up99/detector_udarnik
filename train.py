from ultralytics import YOLO
def train_model():
    # model = YOLO("runs/detect/train3/weights/best.pt")
    model = YOLO("yolov5nu.pt")
    # results = model.train(data="data.yaml", epochs=120, device='cuda', degrees=180.0, patience=50, pretrained = True, augment = True)
    results = model.train(data="data.yaml", epochs=75, device='cuda', patience=15, augment = True, dropout = 0.2, bgr=0.7)
    return results

if __name__ == '__main__':
    # Required for Windows when using PyTorch DataLoader with multiprocessing
    from multiprocessing import freeze_support
    freeze_support()

    train_model()