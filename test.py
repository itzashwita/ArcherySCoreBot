from ultralytics import YOLO

def main():
    # Load trained model
    model = YOLO("runs/archery_yolov88/weights/best.pt")

    # Validate on validation set
    metrics = model.val(data="data.yaml")
    print(metrics)

    # Test on a sample image
    results = model("test/images/sample.jpg", save=True)
    print("Inference complete. Results saved.")

if __name__ == "__main__":
    main()
