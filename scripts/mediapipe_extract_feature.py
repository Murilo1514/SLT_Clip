import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
import tqdm
import os.path as osp
import urllib.request
import multiprocessing as mp_cpu
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from mediapipe.tasks.python.vision.holistic_landmarker import (
        HolisticLandmarker,
        HolisticLandmarkerOptions
    )
    HAS_HOLISTIC = True
except (ImportError, AttributeError):
    HAS_HOLISTIC = False

import sys
sys.path.append('./')
from utils.helpers import sliding_window_for_list, get_img_list, read_video


MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "holistic": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
}


def ensure_model_downloaded(model_name: str, cache_dir: str = "./models") -> str:
    """Garante que o arquivo de modelo (.task) esteja baixado localmente no cache."""
    if model_name not in MODEL_URLS:
        raise ValueError(f"Modelo desconhecido: {model_name}. Escolha entre: {list(MODEL_URLS.keys())}")
    
    os.makedirs(cache_dir, exist_ok=True)
    filename = osp.basename(MODEL_URLS[model_name])
    local_path = osp.join(cache_dir, filename)

    if not osp.exists(local_path) or os.path.getsize(local_path) == 0:
        url = MODEL_URLS[model_name]
        print(f"Baixando modelo '{model_name}' de {url} para {local_path}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Download concluído: {local_path} ({os.path.getsize(local_path) / (1024 * 1024):.2f} MB)")
    
    return local_path


class MediaPipeFeatureReader:
    def __init__(self, model_type="auto", cache_dir="./models", window_size=16, overlap_size=8, flatten_window=True):
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.flatten_window = flatten_window

        if model_type == "auto":
            self.model_type = "holistic" if HAS_HOLISTIC else "separate"
        elif model_type == "holistic" and not HAS_HOLISTIC:
            self.model_type = "separate"
        else:
            self.model_type = model_type

        if self.model_type == "holistic":
            model_path = ensure_model_downloaded("holistic", cache_dir)
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = HolisticLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.holistic_landmarker = HolisticLandmarker.create_from_options(options)
            self.keypoint_dim = (25 + 21 + 21 + 478) * 3
        elif self.model_type == "separate":
            pose_path = ensure_model_downloaded("pose", cache_dir)
            hand_path = ensure_model_downloaded("hand", cache_dir)
            face_path = ensure_model_downloaded("face", cache_dir)

            pose_opts = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.IMAGE,
            )
            hand_opts = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
            )
            face_opts = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )

            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

            self.keypoint_dim = (25 + 21 + 21 + 478) * 3
        else:
            raise ValueError(f"model_type inválido: {self.model_type}")

        self.chunk_dim = self.keypoint_dim * self.window_size if flatten_window else self.keypoint_dim

    def extract_frame_landmarks_holistic(self, mp_image):
        result = self.holistic_landmarker.detect(mp_image)

        if result.pose_landmarks:
            pose_points = result.pose_landmarks[:25]
            pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_points], dtype=np.float32).flatten()
            if len(pose) < 25 * 3:
                pose = np.pad(pose, (0, 25 * 3 - len(pose)))
        else:
            pose = np.zeros(25 * 3, dtype=np.float32)

        if result.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks], dtype=np.float32).flatten()
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        if result.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks], dtype=np.float32).flatten()
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        if result.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks], dtype=np.float32).flatten()
            if len(face) < 478 * 3:
                face = np.pad(face, (0, 478 * 3 - len(face)))
            elif len(face) > 478 * 3:
                face = face[:478 * 3]
        else:
            face = np.zeros(478 * 3, dtype=np.float32)

        return np.concatenate([pose, lh, rh, face])

    def extract_frame_landmarks_separate(self, mp_image):
        pose_res = self.pose_landmarker.detect(mp_image)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            pose_points = pose_res.pose_landmarks[0][:25]
            pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_points], dtype=np.float32).flatten()
            if len(pose) < 25 * 3:
                pose = np.pad(pose, (0, 25 * 3 - len(pose)))
        else:
            pose = np.zeros(25 * 3, dtype=np.float32)

        lh = np.zeros(21 * 3, dtype=np.float32)
        rh = np.zeros(21 * 3, dtype=np.float32)
        hand_res = self.hand_landmarker.detect(mp_image)
        if hand_res.hand_landmarks and hand_res.handedness:
            for hand_landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = handedness[0].category_name
                kps = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32).flatten()
                if label == 'Left':
                    lh = kps
                elif label == 'Right':
                    rh = kps
 
        face_res = self.face_landmarker.detect(mp_image)
        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            face = np.array([[lm.x, lm.y, lm.z] for lm in face_res.face_landmarks[0]], dtype=np.float32).flatten()
            if len(face) < 478 * 3:
                face = np.pad(face, (0, 478 * 3 - len(face)))
            elif len(face) > 478 * 3:
                face = face[:478 * 3]
        else:
            face = np.zeros(478 * 3, dtype=np.float32)

        return np.concatenate([pose, lh, rh, face])

    def extract_frame_landmarks(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.model_type == "holistic":
            return self.extract_frame_landmarks_holistic(mp_image)
        else:
            return self.extract_frame_landmarks_separate(mp_image)

    def get_feats(self, video_frames):
        raw_kps = []
        for img in video_frames:
            if isinstance(img, str):
                bgr = cv2.imread(img)
                if bgr is None:
                    continue
                frame_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif isinstance(img, Image.Image):
                frame_np = np.array(img)
            elif isinstance(img, np.ndarray):
                frame_np = img
            else:
                continue

            kps = self.extract_frame_landmarks(frame_np)
            raw_kps.append(kps)

        if len(raw_kps) == 0:
            return np.empty((0, self.chunk_dim), dtype=np.float32)

        if len(raw_kps) < self.window_size:
            raw_kps.extend([raw_kps[-1]] * (self.window_size - len(raw_kps)))

        chunks = sliding_window_for_list(raw_kps, window_size=self.window_size, overlap_size=self.overlap_size)
        chunks_np = np.array(chunks, dtype=np.float32)

        if self.flatten_window:
            num_chunks = chunks_np.shape[0]
            chunks_np = chunks_np.reshape(num_chunks, -1)

        return chunks_np


def get_iterator(args, mode, reader, indices=None):
    """
    Mantém o padrão original de iteração de datasets do SpaMo (Phoenix14T, CSL-Daily, How2Sign).
    Se `indices` for fornecido, itera sobre o subconjunto atribuído ao worker.
    """
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    total_num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    target_indices = indices if indices is not None else range(total_num)
    count = len(target_indices)

    def iterate():
        for i in target_indices:
            fname = data[i]['folder']
            file_id = data[i]['fileid']
            
            if ds_name in ['Phoenix14T', 'CSL-Daily']:
                image_list = get_img_list(ds_name, args.video_root, fname)
                if len(image_list) == 0:
                    yield np.empty((0, reader.chunk_dim), dtype=np.float32), file_id, None
                    continue
                # Lê os frames usando a lista de imagens
                videos = [Image.open(img).convert('RGB') for img in image_list]
                feats = reader.get_feats(videos)
                yield feats, file_id, None

            elif ds_name == 'How2Sign':
                start_time = data[i]['original_info']['START_REALIGNED']
                end_time = data[i]['original_info']['END_REALIGNED']
                videos = read_video(fname, start_time=start_time, end_time=end_time)
                if len(videos) > 0:
                    feats = reader.get_feats(videos)
                    yield feats, file_id, str(start_time)
                else:
                    yield np.empty((0, reader.chunk_dim), dtype=np.float32), file_id, str(start_time)
    
    return iterate, count


def worker_task(worker_id, sample_indices, mode_name, args, save_path):
    """Executado por cada worker independente em paralelo."""
    cv2.setNumThreads(1)

    reader = MediaPipeFeatureReader(
        model_type=args.model_type,
        cache_dir=args.cache_dir,
        window_size=args.window_size,
        overlap_size=args.overlap_size
    )

    generator, total_samples = get_iterator(args, mode_name, reader, indices=sample_indices)

    iterator = generator()
    if worker_id == 0:
        iterator = tqdm.tqdm(iterator, total=total_samples, desc=f"Split {mode_name} ({args.num_workers} workers)")

    for feats, file_id, st in iterator:
        postfix = f'_overlap-{args.overlap_size}-mediapipe'
        if st is not None:
            postfix = f'_{st}{postfix}'

        out_file = osp.join(save_path, f'{file_id}{postfix}.npy')
        np.save(out_file, feats)


def get_parser():
    parser = argparse.ArgumentParser(description="Extração Multiprocessamento de Keypoints (MediaPipe) para o SpaMo")
    parser.add_argument('--anno_root', help='Caminho para anotações (ex: ./preprocess/Phoenix14T)', required=True)
    parser.add_argument('--video_root', help='Caminho base dos vídeos/frames', required=True)
    parser.add_argument('--save_dir', help='Diretório onde salvar as features extraídas', required=True)
    parser.add_argument('--model_type', type=str, default='auto', choices=['auto', 'holistic', 'separate'],
                        help="Modelo a utilizar: 'auto', 'holistic' ou 'separate'")
    parser.add_argument('--cache_dir', type=str, default='./models', help='Diretório dos modelos .task')
    parser.add_argument('--num_workers', type=int, default=28, help='Número de processos de CPU paralelos')
    parser.add_argument('--window_size', type=int, default=16, help='Tamanho da janela temporal (padrão: 16)')
    parser.add_argument('--overlap_size', type=int, default=8, help='Tamanho do overlap temporal (padrão: 8)')
    parser.add_argument('--mode', nargs='+', default=['dev', 'test', 'train'], help='Splits para processar')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Garante download prévio dos modelos antes de disparar os workers
    if args.model_type in ["auto", "holistic"] and HAS_HOLISTIC:
        ensure_model_downloaded("holistic", args.cache_dir)
    else:
        ensure_model_downloaded("pose", args.cache_dir)
        ensure_model_downloaded("hand", args.cache_dir)
        ensure_model_downloaded("face", args.cache_dir)

    ds_name = osp.split(args.anno_root)[-1]
    folder_name = f'mediapipe_feat_{ds_name}'
    num_workers = min(args.num_workers, mp_cpu.cpu_count())

    for m in args.mode:
        print(f"\n=======================================================")
        print(f"--- Iniciando extração do split: {m} ({ds_name}) com {num_workers} workers ---")
        print(f"=======================================================")
        save_path = osp.join(args.save_dir, folder_name, m)
        os.makedirs(save_path, exist_ok=True)

        if ds_name == 'How2Sign':
            _m = 'val' if m == 'dev' else m
        elif ds_name == 'NIASL2021':
            _m = 'validation' if m == 'dev' else m
        else:
            _m = m

        # Carrega tamanho do split para particionar entre workers
        data = np.load(os.path.join(args.anno_root, f'{_m}_info.npy'), allow_pickle=True).item()
        total_samples = len(data) - 1
        all_indices = list(range(total_samples))

        if num_workers <= 1:
            # Modo single-process tradicional
            reader = MediaPipeFeatureReader(
                model_type=args.model_type,
                cache_dir=args.cache_dir,
                window_size=args.window_size,
                overlap_size=args.overlap_size
            )
            generator, count = get_iterator(args, _m, reader)
            for feats, file_id, st in tqdm.tqdm(generator(), total=count, desc=f"Split {m}"):
                postfix = f'_overlap-{args.overlap_size}-mediapipe'
                if st is not None:
                    postfix = f'_{st}{postfix}'
                out_file = osp.join(save_path, f'{file_id}{postfix}.npy')
                np.save(out_file, feats)
        else:
            # Modo multi-process: divide os índices e roda workers usando get_iterator
            split_chunks = np.array_split(all_indices, num_workers)
            processes = []
            for worker_id in range(num_workers):
                indices_for_worker = split_chunks[worker_id].tolist()
                if len(indices_for_worker) == 0:
                    continue

                p = mp_cpu.Process(
                    target=worker_task,
                    args=(worker_id, indices_for_worker, _m, args, save_path)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

    print(f"\nExtração concluída com sucesso! Features salvas em: {osp.join(args.save_dir, folder_name)}")


if __name__ == "__main__":
    main()
