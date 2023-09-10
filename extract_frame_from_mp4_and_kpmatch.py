import cv2
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d



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


def integrate(input_file, start_frame, end_frame, frame_interval, output_directory,output_csv_path,output_video_name, threshold=0):
    # MP4ファイルを開く
    cap = cv2.VideoCapture(input_file)
    # フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    # 開始フレームと終了フレームを調整
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_directory, exist_ok=True)
    # フレーム間隔でフレームを抽出し保存
    current_frame = start_frame
    frame_count = 0
    while current_frame <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        # フレームの時間を取得
        frame_time = current_frame / cap.get(cv2.CAP_PROP_FPS)
        # 画像の保存
        output_file = os.path.join(output_directory, f"{frame_time:07.2f}.png")
        cv2.imwrite(output_file, frame)

        frame_count += 1
        current_frame += frame_interval

    # ディレクトリ内のファイルをリストアップ
    file_list = os.listdir(output_directory)
    
    # ファイルをソートして順番に処理
    file_list.sort()
    
    # ファイル数が1以下の場合は処理不可
    if len(file_list) <= 1:
        print("処理対象の画像ファイルが足りません。")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_name, fourcc, frame_rate/frame_interval, (int(cap.get(3)), int(cap.get(4))))

    # 特徴点抽出
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
    matcher = LightGlue(features='superpoint').eval().to(device)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    total_movements = []
    # 画像ファイルに対する処理を順番に実行
    for i in range(len(file_list) - 1):
        image_path0 = os.path.join(output_directory, file_list[i])
        image_path1 = os.path.join(output_directory, file_list[i + 1])
        print(f"処理中: {image_path0}, {image_path1}")
          
        image0 = load_image(image_path0) 
        image1 = load_image(image_path1)
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        p0 = m_kpts0.numpy().astype(np.float32).reshape(-1, 1, 2)
        p1 = m_kpts1.numpy().astype(np.float32).reshape(-1, 1, 2)
        flow = p1 - p0
        # 各点の移動量を計算
        magnitudes = np.linalg.norm(flow, axis=2).squeeze() #各点の移動量
        significant_movements = magnitudes[magnitudes >= threshold] #移動量が閾値以上のものを抽出
        total_movement = np.sum(significant_movements) #そのフレームでの移動量の総和計算
        average_movement = total_movement / len(magnitudes) #移動量の平均計算
        total_movements.append(average_movement)
        # オプティカルフローの視覚化
        frame = cv2.imread(image_path1)
        vis_frame = draw_flow(frame, p0, flow)
        out.write(vis_frame)
        # cv2.imshow('kpmatch Flow', vis_frame)

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
                writer.writerow([i*frame_interval/frame_rate, movement])

def plot(plot_csv_name, output_chart_name):
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
    plt.xlim(0, max(time))
    plt.ylim(0, max(average_movements))
    # データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
    plt.plot(time, average_movements, label='Time waveform', lw=1, color='red')
    # レイアウト設定
    plt.tight_layout()

    # グラフを保存する
    plt.savefig(output_chart_name)
    plt.close()
    # ---------------------------------------------------


if __name__ == "__main__":
    input_file = 'assets/optical_flow_test5.mp4'
    start_frame = 0
    end_frame = 100
    frame_interval = 10
    current_time = datetime.datetime.now()
    output_directory = 'kpmatch_frames/' + current_time.strftime('%Y-%m-%d_%H-%M-%S')
    output_csv_name = "kpmatch_csv/" + current_time.strftime('%Y-%m-%d_%H-%M-%S.csv')
    output_video_name = "kpmatch_video/" + current_time.strftime('%Y-%m-%d_%H-%M-%S.avi')
    output_chart_name = "kpmatch_charts/" + current_time.strftime('%Y-%m-%d_%H-%M-%S.png') 

    integrate(input_file, start_frame, end_frame, frame_interval, output_directory, output_csv_name,output_video_name, threshold=0)

    plot(output_csv_name, output_chart_name)
