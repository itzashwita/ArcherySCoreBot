from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,          # lower batch for CPU
        device="cpu",
        project="runs",
        name="archery_yolov8"
    )

if __name__ == "__main__":
    main()
