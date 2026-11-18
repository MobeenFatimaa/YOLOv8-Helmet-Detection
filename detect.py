import cv2
from ultralytics import YOLO

def main():
    # 1. Load your trained model weights
    model = YOLO("runs/detect/helmet_custom_model-5/weights/best.pt")

    input_video_path = "test_video.mp4"
    output_video_path = "output_detected.mp4"

    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}. Make sure 'test_video.mp4' is in your project folder.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video and saving to '{output_video_path}'... Please wait.")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference without GUI pop-up windows
        results = model(frame, conf=0.5, verbose=False)

        # Plot bounding boxes onto the frame
        annotated_frame = results[0].plot()

        # Write frame to output video
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # Clean up
    cap.release()
    out.release()
    print(f"Success! Video processing complete. Saved as '{output_video_path}'.")

if __name__ == "__main__":
    main()
