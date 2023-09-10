import cv2
import numpy as np
import torch
from pathlib import Path
import math
import datetime
import matplotlib.pyplot as plt

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d

torch.set_grad_enabled(False)
images = Path('assets')

def draw_flow(img, points, flow, color=(0, 255, 0)):
    """
    指定した画像における指定した点にフローを描画する関数
    
    Parameters:
    - image: 画像ファイルへのパス
    - points: フローを描画する点
    - fllow: 描画するフロー
    - color: フローの色
    """
    for i, (point, flow_vec) in enumerate(zip(points, flow)):
        x, y = point.ravel()
        dx, dy = flow_vec.ravel()
        end = (int(x+dx), int(y+dy))
        img = cv2.line(img, (int(x), int(y)), end, color, 2)
        img = cv2.circle(img, end, 5, color, -1)
    return img

def compute_and_visualize_optical_flow(video_path, points_tensor, start_frame, end_frame, output_csv_path, threshold=0, frame_interval=1):
    """
    指定した動画における指定した点のオプティカルフローを計算する関数
    
    Parameters:
    - video_path: 動画ファイルのパス
    - points_tensor: オプティカルフローを計算したい点
    - start_frame: 開始フレーム
    - end_frame: 終了フレーム
    - output_csv_path: 移動量の平均を記録するcsvファイルのパス
    - threshold: 移動量の平均を計算する際、移動量が一定以下の点は除外するための閾値
    - frame_interval: オプティカルフローを計算するフレームの間隔
    """
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
    
    # 初期化
    total_movements = []
    interval_count = -1
    average_movement = 0

    # フレームごとに処理
    while True:
        # interval_frameで指定した間隔でフレーム読込
        interval_count += 1
        if interval_count % frame_interval == 0:
            ret, frame = cap.read()
            interval_count = 0
        else:
            cap.read()
        #読み込むフレームがなくなったら終了
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break

        # フレームをグレスケ化
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # オプティカルフローを計算
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        
        valid_points = st == 1
        if np.any(valid_points):
            # トラッキングが成功した点のみで処理を続行
            p0 = p0[valid_points].reshape(-1, 1, 2)
            p1 = p1[valid_points].reshape(-1, 1, 2)
            flow = p1 - p0
            # 各点の移動量を計算
            magnitudes = np.linalg.norm(flow, axis=2).squeeze() #各点の移動量
            significant_movements = magnitudes[magnitudes >= threshold] #移動量が閾値以上のものを抽出
            total_movement = np.sum(significant_movements) #そのフレームでの移動量の総和計算
            if interval_count % frame_interval == 0:
                average_movement = total_movement / len(magnitudes) #移動量の平均計算
            total_movements.append(average_movement)
            # オプティカルフローの視覚化
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
        
    # 各フレーム間での移動量の平均をcsv形式で保存
    if output_csv_path:
        import csv
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Total Movement"])
            for i, movement in enumerate(total_movements):
                writer.writerow([(i + start_frame)/frame_rate, movement])

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
    # フレームが読み込めなかったときの処理
    if not ret:
        print(f"Error: Could not read frame {frame_number} from video.")
        return None
    # 結果の出力パス 
    if not output_name:
        output_name = f'frame_{frame_number}.png'
    output_path = os.path.join(output_dir, output_name)
    # 出力パスに画像つくりメモリ開放してパスを返す
    cv2.imwrite(output_path, frame)
    cap.release()
    return output_path

def plot(plot_csv_name):
    # ここからグラフ描画-------------------------------------
    data = np.loadtxt(plot_csv_name, usecols=(0, 1), delimiter=',',skiprows=1,  encoding="utf-8_sig")
    time = data[:, 0]
    average_movements = data[:, 1]
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # グラフの上下左右に目盛線を付ける。
    # fig = plt.figure()
    # 軸のラベルを設定する。
    plt.xlabel('Time [s]')
    plt.ylabel('average_movements')
    # スケールの設定をする。
    plt.xlim(0, 10)
    plt.ylim(0, max(average_movements))
    # データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
    plt.plot(time, average_movements, label='Time waveform', lw=1, color='red')
    # レイアウト設定
    plt.tight_layout()

    # グラフを保存する
    output_chart_name = "flow_charts/" + current_time.strftime('%Y-%m-%d_%H-%M-%S.png')
    plt.savefig(output_chart_name)
    plt.close()
    # ---------------------------------------------------

if __name__ == "__main__":

    # オプティカルフローを計算する動画ファイルの指定
    video_path = 'assets/optical_flow_test4.mp4'
    start_frame = 1
    end_frame = 1000
    
    # 特徴点抽出を行うフレームの指定
    firts_frame = save_frame_as_image(video_path, frame_number=1, output_dir="assets/", output_name="frame_first.png")
    # 特徴点抽出
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

    # オプティカルフローの計算と描画
    current_time = datetime.datetime.now()
    output_csv_name = "flow_csv/" + current_time.strftime('%Y-%m-%d_%H-%M-%S.csv')
    compute_and_visualize_optical_flow(video_path, kpts0, start_frame, end_frame, output_csv_name, threshold=0, frame_interval=10)

    plot(output_csv_name)

    