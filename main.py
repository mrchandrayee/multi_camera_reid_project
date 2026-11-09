from __future__ import annotations
import argparse, os, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import MultiCameraTracker

def parse_sources(values):
    sources = {}
    for i, value in enumerate(values):
        # Syntax: CAM_NAME=path_or_camera_index
        if "=" in value:
            name, src = value.split("=", 1)
        else:
            name, src = f"CAM-{i+1:02d}", value
        sources[name] = int(src) if src.isdigit() else src
    return sources

def main():
    ap = argparse.ArgumentParser(description="Multi-camera person detection and ReID demo")
    ap.add_argument("--source", nargs="+", required=True,
                    help="One or more sources: CAM1=video.mp4 CAM2=0")
    ap.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    ap.add_argument("--confidence", type=float, default=0.35)
    ap.add_argument("--similarity", type=float, default=0.72,
                    help="Cross-camera embedding similarity threshold")
    ap.add_argument("--display-width", type=int, default=960)
    ap.add_argument("--save-dir", default="runs/output")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    sources = parse_sources(args.source)
    model = YOLO(args.model)  # downloads weights on first run if not present
    tracker = MultiCameraTracker(similarity_threshold=args.similarity)
    os.makedirs(args.save_dir, exist_ok=True)

    captures = {}
    writers = {}
    for name, src in sources.items():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[WARN] Could not open {name}: {src}")
            continue
        captures[name] = cap
        if not args.no_display:
            print(f"[INFO] Opened {name}: {src}")

    if not captures:
        raise SystemExit("No video sources could be opened.")

    try:
        while captures:
            active = False
            for cam_name, cap in list(captures.items()):
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    captures.pop(cam_name)
                    if cam_name in writers:
                        writers.pop(cam_name).release()
                    continue
                active = True
                result = model.predict(frame, conf=args.confidence, classes=[0],
                                       verbose=False, device=None)[0]
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(box.conf[0].cpu())
                        x1,y1,x2,y2 = xyxy
                        if x2 > x1 and y2 > y1:
                            detections.append((xyxy, conf))
                tracks = tracker.update(cam_name, frame, detections)
                output = tracker.draw(frame, cam_name, tracks)
                h,w = output.shape[:2]
                if w > args.display_width:
                    output = cv2.resize(output, (args.display_width, int(h*args.display_width/w)))
                if not args.no_display:
                    cv2.imshow(f"Person ReID - {cam_name}", output)
                out_path = os.path.join(args.save_dir, f"{cam_name}.mp4")
                if cam_name not in writers:
                    oh,ow = output.shape[:2]
                    writers[cam_name] = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (ow,oh))
                writers[cam_name].write(output)
            if not active:
                break
            if not args.no_display and (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        for cap in captures.values():
            cap.release()
        for writer in writers.values():
            writer.release()
        cv2.destroyAllWindows()
    print(f"[DONE] Annotated videos saved in: {args.save_dir}")

if __name__ == "__main__":
    main()
