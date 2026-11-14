import os
import cv2
import numpy as np
import argparse
import tqdm
import os.path as osp
from PIL import Image

from rtmlib import Wholebody

import sys
sys.path.append('./')
from utils.helpers import sliding_window_for_list, get_img_list, read_video


class RTMPoseFeatureReader:
    def __init__(
        self,
        mode='balanced',
        backend='onnxruntime',
        device='cuda',
        to_openpose=False,
        save_scores=True,
        window_size=16,
        overlap_size=8,
        use_sliding_window=False,
        flatten_window=True
    ):
        """
        Extrator de pose e landmarks corporais completos usando RTMPose / Wholebody via rtmlib.
        
        Extrai 133 pontos do COCO-Wholebody:
          - 17 pontos de pose (corpo)
          - 6 pontos de pés
          - 68 pontos de face
          - 21 pontos mão esquerda
          - 21 pontos mão direita
        """
        self.device = device
        self.save_scores = save_scores
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.flatten_window = flatten_window

        print(f"Inicializando RTMPose (mode='{mode}', backend='{backend}', device='{device}')...")
        self.model = Wholebody(
            to_openpose=to_openpose,
            mode=mode,
            backend=backend,
            device=device
        )
        print("Modelo RTMPose carregado com sucesso!")

    def extract_frame_landmarks(self, frame_bgr):
        """
        Executa inferência em um único frame BGR (formato nativo do OpenCV).
        Retorna:
          - se save_scores=True: array shape (133, 3) com [x, y, score]
          - se save_scores=False: array shape (133, 2) com [x, y]
        """
        keypoints, scores = self.model(frame_bgr)
        # keypoints shape: (N, 133, 2), scores shape: (N, 133)
        if len(keypoints) > 0:
            # Seleciona a primeira pessoa detectada (em SLT geralmente há 1 sinalizador em foco)
            kpts = keypoints[0]  # (133, 2)
            scs = scores[0]      # (133,)
        else:
            kpts = np.zeros((133, 2), dtype=np.float32)
            scs = np.zeros((133,), dtype=np.float32)

        if self.save_scores:
            # Concatena score como 3ª dimensão: (133, 3)
            return np.hstack([kpts, scs[:, np.newaxis]]).astype(np.float32)
        else:
            return kpts.astype(np.float32)

    def get_feats(self, frame_paths_or_images):
        """
        Recebe uma lista de caminhos de imagens (strings) ou lista de PIL/numpy frames.
        Extrai os keypoints para cada frame do vídeo.
        
        Retorna:
          - Sem sliding window: array shape (T, 133, 3) onde T = número de frames
          - Com sliding window: array shape (num_chunks, 16 * 133 * 3) se flatten_window=True
        """
        raw_kps = []
        for item in frame_paths_or_images:
            if isinstance(item, str):
                frame_bgr = cv2.imread(item)
            elif isinstance(item, Image.Image):
                frame_bgr = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
            elif isinstance(item, np.ndarray):
                # Se for RGB de 3 canais, converte para BGR
                frame_bgr = cv2.cvtColor(item, cv2.COLOR_RGB2BGR) if item.ndim == 3 else item
            else:
                continue

            if frame_bgr is None:
                continue

            kps = self.extract_frame_landmarks(frame_bgr)
            raw_kps.append(kps)

        if len(raw_kps) == 0:
            dim = 3 if self.save_scores else 2
            return np.empty((0, 133, dim), dtype=np.float32)

        if not self.use_sliding_window:
            # Retorna sequência temporal direta de keypoints: (T, 133, 3)
            return np.array(raw_kps, dtype=np.float32)

        # Caso deseje aplicar janelamento deslizante:
        if len(raw_kps) < self.window_size:
            raw_kps.extend([raw_kps[-1]] * (self.window_size - len(raw_kps)))

        # Achata os keypoints por frame antes de passar para a janela
        flat_kps = [k.flatten() for k in raw_kps]
        chunks = sliding_window_for_list(flat_kps, window_size=self.window_size, overlap_size=self.overlap_size)
        chunks_np = np.array(chunks, dtype=np.float32)

        if self.flatten_window:
            num_chunks = chunks_np.shape[0]
            chunks_np = chunks_np.reshape(num_chunks, -1)

        return chunks_np


def get_parser():
    parser = argparse.ArgumentParser(description="Extração de Whole-body Keypoints com RTMPose (rtmlib)")
    parser.add_argument('--anno_root', help='Caminho para anotações (ex: ./preprocess/Phoenix14T)', required=True)
    parser.add_argument('--video_root', help='Caminho base dos vídeos/frames', required=True)
    parser.add_argument('--save_dir', help='Diretório onde salvar as features extraídas', required=True)
    parser.add_argument('--device', type=str, default='cuda:0', help="Dispositivo para execução ('cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument('--mode_rtm', type=str, default='balanced', choices=['performance', 'balanced', 'lightweight'],
                        help="Variante do modelo RTMPose: 'performance' (mais preciso), 'balanced' (recomendado), 'lightweight' (mais rápido)")
    parser.add_argument('--backend', type=str, default='onnxruntime', choices=['onnxruntime', 'opencv', 'openvino'],
                        help="Backend de inferência")
    parser.add_argument('--use_sliding_window', action='store_true',
                        help="Se ativado, aplica janelamento deslizante (window_size=16, overlap=8). Se desativado, salva array (T, 133, 3).")
    parser.add_argument('--window_size', type=int, default=16, help='Tamanho da janela temporal (padrão: 16)')
    parser.add_argument('--overlap_size', type=int, default=8, help='Tamanho do overlap temporal (padrão: 8)')
    parser.add_argument('--mode', nargs='+', default=['dev', 'test', 'train'], help='Splits para processar')
    return parser


def get_iterator(args, mode, reader):
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    def iterate():
        for i in range(num):
            fname = data[i]['folder']
            file_id = data[i]['fileid']

            if ds_name in ['Phoenix14T', 'CSL-Daily']:
                image_list = get_img_list(ds_name, args.video_root, fname)
                if len(image_list) == 0:
                    yield np.empty((0, 133, 3), dtype=np.float32), file_id, None
                    continue
                # Passa caminhos diretos para cv2.imread acelerado
                feats = reader.get_feats(image_list)
                yield feats, file_id, None

            elif ds_name == 'How2Sign':
                start_time = data[i]['original_info']['START_REALIGNED']
                end_time = data[i]['original_info']['END_REALIGNED']
                videos = read_video(fname, start_time=start_time, end_time=end_time)
                if len(videos) > 0:
                    feats = reader.get_feats(videos)
                    yield feats, file_id, str(start_time)
                else:
                    yield np.empty((0, 133, 3), dtype=np.float32), file_id, str(start_time)

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    ds_name = osp.split(args.anno_root)[-1]
    folder_name = f'rtmpose_feat_{ds_name}'

    reader = RTMPoseFeatureReader(
        mode=args.mode_rtm,
        backend=args.backend,
        device=args.device,
        save_scores=True,
        use_sliding_window=args.use_sliding_window,
        window_size=args.window_size,
        overlap_size=args.overlap_size
    )

    for m in args.mode:
        print(f"\n--- Iniciando extração do split: {m} ({ds_name}) ---")
        save_path = osp.join(args.save_dir, folder_name, m)
        os.makedirs(save_path, exist_ok=True)

        if ds_name == 'How2Sign':
            _m = 'val' if m == 'dev' else m
        elif ds_name == 'NIASL2021':
            _m = 'validation' if m == 'dev' else m
        else:
            _m = m

        generator, total_samples = get_iterator(args, _m, reader)
        for feats, file_id, st in tqdm.tqdm(generator(), total=total_samples, desc=f"Split {m}"):
            postfix = '_rtmpose'
            if args.use_sliding_window:
                postfix = f'_overlap-{args.overlap_size}-rtmpose'
            if st is not None:
                postfix = f'_{st}{postfix}'

            out_file = osp.join(save_path, f'{file_id}{postfix}.npy')
            np.save(out_file, feats)

    print(f"\nExtração concluída com sucesso! Features salvas em: {osp.join(args.save_dir, folder_name)}")


if __name__ == "__main__":
    main()

