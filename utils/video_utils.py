import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def save_video(output_video_frames, output_video_path, fps):
    if not output_video_frames:
        print("Error: No frames to save!")
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved successfully: {output_video_path}")