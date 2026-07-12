from ultralytics import YOLO

def main():
    print("Loading YOLO11n model...")
    # This will automatically download yolo11n.pt if not present
    model = YOLO("yolo11n.pt")
    
    print("Exporting model to TensorRT engine (FP16)...")
    print("NOTE: You must have NVIDIA GPU and TensorRT installed on your system.")
    try:
        # Export the model
        # 'half=True' ensures FP16 precision for faster inference.
        model.export(format="engine", half=True)
        print("Export complete! You can now use yolo11n.engine for anti-delay inference.")
    except Exception as e:
        print(f"Failed to export to TensorRT: {e}")
        print("Exporting to ONNX as fallback...")
        model.export(format="onnx")

if __name__ == "__main__":
    main()
