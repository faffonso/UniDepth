import numpy as np
import torch
from PIL import Image
import cv2
from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.utils import colorize

def process_frame(model, frame):
    # Convert frame to RGB format expected by the model
    rgb_torch = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    rgb_torch = rgb_torch.unsqueeze(0).to(next(model.parameters()).device)

    # Predict depth
    predictions = model.infer(rgb_torch)

    # Extract depth prediction
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

    # Colorize the depth map for visualization
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
    return depth_pred_col

def main(model, input_video_path, output_video_path, X):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Set up the video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    frame_idx = 0
    cached_depth_pred = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Process frame only every X frames
        if frame_idx % X == 1 or cached_depth_pred is None:
            print(f"Processing frame {frame_idx}/{frame_count}...")
            # Convert BGR to RGB for the model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate depth prediction
            cached_depth_pred = process_frame(model, frame_rgb)

        # Repeat the last processed depth prediction
        depth_pred_bgr = cv2.cvtColor(cached_depth_pred, cv2.COLOR_RGB2BGR)

        # Write the repeated frame to the output video
        out.write(depth_pred_bgr)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    name = "unidepth-v2-vitl14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_video = "../assets/demo/video.mp4"  # Replace with your input video path
    output_video = "../assets/demo/output.mp4"  # Replace with your desired output path

    frame_interval = 200  # Replace with your desired interval
    main(model, input_video, output_video, frame_interval)

