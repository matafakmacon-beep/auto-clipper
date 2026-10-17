import cv2
import numpy as np
import subprocess
import os

def run_auto_clipper(video_path, template_path, roi_box, threshold=0.78, padding_before=1.75, padding_after=1.25):
    if not os.path.exists(video_path):
        print(f"❌ ไม่พบไฟล์วิดีโอ: {video_path}")
        return

    if not os.path.exists(template_path):
        print(f"❌ ไม่พบไฟล์รูปภาพอ้างอิง: {template_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    th, tw = template.shape[:2]
    
    rx, ry, rw, rh = roi_box
    timestamps = []
    frame_idx = 0

    print(f"🚀 กำลังเริ่มสแกนวิดีโอ (ความยาวทั้งหมดประมาณ {total_frames / fps:.2f} วินาที)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[ry:ry+rh, rx:rx+rw]
        
        if roi_frame.shape[0] < th or roi_frame.shape[1] < tw:
            break

        res = cv2.matchTemplate(roi_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            time_sec = frame_idx / fps
            timestamps.append(time_sec)
            print(f"🎯 พบคิลที่วินาทีที่: {time_sec:.2f} (ความแม่นยำ: {max_val:.2f})")
            
            skip_frames = int(fps * (padding_after + 1.5))
            frame_idx += skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            continue

        frame_idx += 1

    cap.release()
    print(f"✨ สแกนเสร็จสิ้น! พบไฮไลท์ทั้งหมด {len(timestamps)} จุด")

    if timestamps:
        export_clips(video_path, timestamps, padding_before, padding_after)

def export_clips(video_path, timestamps, padding_before, padding_after, output_dir="output_highlights"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("🎬 กำลังตัดต่อและบันทึกคลิป...")
    for idx, t in enumerate(timestamps):
        start_time = max(0, t - padding_before)
        duration = padding_before + padding_after
        output_file = os.path.join(output_dir, f"highlight_{idx+1}.mp4")

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c', 'copy',
            output_file
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"💾 บันทึกสำเร็จ: {output_file}")

if __name__ == "__main__":
    VIDEO_FILE = "gameplay.mp4"
    TEMPLATE_FILE = "kill_icon.png"
    ROI_REGION = (500, 150, 120, 120) 

    run_auto_clipper(
        video_path=VIDEO_FILE,
        template_path=TEMPLATE_FILE,
        roi_box=ROI_REGION,
        threshold=0.78,
        padding_before=1.75,
        padding_after=1.25
    )
