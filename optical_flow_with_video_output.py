import cv2
import numpy as np
import torch
from pathlib import Path
    
from lightglue import LightGlue, SuperPoint, DISK
# import SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import math
import datetime
torch.set_grad_enabled(False)
images = Path('assets')

def draw_flow(img, points, flow, color=(0, 255, 0)):
    for i, (point, flow_vec) in enumerate(zip(points, flow)):
        x, y = point.ravel()
        dx, dy = flow_vec.ravel()
        end = (int(x+dx), int(y+dy))
        img = cv2.line(img, (int(x), int(y)), end, color, 2)
        img = cv2.circle(img, end, 5, color, -1)
    return img

def compute_and_visualize_optical_flow(video_path, points_tensor, start_frame, end_frame, output_csv_path, threshold=0, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_optical_flow.avi', fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

    
    # 指定したフレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Tensorオブジェクトをnumpy.arrayに変換
    p0 = points_tensor.numpy().astype(np.float32).reshape(-1, 1, 2)
    
    # Initialize a list to store total_movement for each frame
    total_movements = []
    count = -1
    average_movement = 0

    while True:
        count += 1
        if count % frame_interval == 0:
            ret, frame = cap.read()
            count = 0
        else:
            cap.read()
        
        # for i in range(frame_interval):
        #     ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # オプティカルフローを計算
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        
        valid_points = st == 1
        if np.any(valid_points):  # if any tracking is successful
            
            # Use only valid points and flows for visualization and for the next frame
            p0 = p0[valid_points].reshape(-1, 1, 2)
            p1 = p1[valid_points].reshape(-1, 1, 2)
            flow = p1 - p0
            # Compute the magnitude of the movement for each point
            magnitudes = np.linalg.norm(flow, axis=2).squeeze()
            significant_movements = magnitudes[magnitudes >= threshold]
            total_movement = np.sum(significant_movements)
            if count % frame_interval == 0:
                average_movement = total_movement / len(significant_movements)
            # Append the total movement for this frame to the list
            total_movements.append(average_movement)

            vis_frame = draw_flow(frame.copy(), p0, flow)
            out.write(vis_frame)
            cv2.imshow('Optical Flow', vis_frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # ESCキーで終了
                break
            p0 = p1
            old_gray = frame_gray.copy()
        else:
            print("All tracking failed!")
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()
        
        # Save the total_movements to a CSV file if an output path is specified
    if output_csv_path:
        import csv
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Total Movement"])
            for i, movement in enumerate(total_movements):
                writer.writerow([(i + start_frame)/frame_rate, movement])
    return total_movement

import cv2
import os

def save_frame_as_image(video_path, frame_number, output_dir='./', output_name=None):
    """
    指定した動画の指定したフレームを画像として保存する関数。
    
    Parameters:
    - video_path: 動画ファイルのパス
    - frame_number: 保存したいフレームの番号
    - output_dir: 保存先のディレクトリ（デフォルトは現在のディレクトリ）
    - output_name: 保存する画像のファイル名（指定しない場合、'frame_{frame_number}.png'となる）

    Returns:
    - 保存した画像ファイルへのパス
    """
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # フレーム番号が動画の範囲内にあるかチェック
    if frame_number >= total_frames or frame_number < 0:
        print(f"Error: Frame number {frame_number} is out of bounds for video with {total_frames} frames.")
        return None

    # 指定したフレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number} from video.")
        return None
    
    if not output_name:
        output_name = f'frame_{frame_number}.png'
    
    output_path = os.path.join(output_dir, output_name)
    cv2.imwrite(output_path, frame)

    cap.release()

    
    return output_path

if __name__ == "__main__":

    # 使用例
    video_path = 'assets/optical_flow_test2.mp4'
    firts_frame = save_frame_as_image(video_path, frame_number=1, output_dir="assets/", output_name="first.png")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
    matcher = LightGlue(features='superpoint').eval().to(device)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)   

    image0 = load_image(firts_frame) 
    image1 = load_image(firts_frame)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']

    start_frame = 1
    end_frame = 1000

    # 現在の日時を取得
    current_time = datetime.datetime.now()
    # 日時を文字列に変換してファイル名を作成
    file_name = current_time.strftime('%Y-%m-%d_%H-%M-%S.csv')
    compute_and_visualize_optical_flow(video_path, kpts0, start_frame, end_frame, file_name, threshold=0, frame_interval=10)
