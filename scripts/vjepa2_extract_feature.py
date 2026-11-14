"""
V-JEPA 2.1 Feature Extractor for SpaMo.

Extracts spatio-temporal video features using Meta's V-JEPA 2.1 (ViT encoder),
following the same offline extraction pipeline as mae_extract_feature.py.

Key differences from VideoMAE:
  - V-JEPA 2.1 expects 64 frames per clip (fpc64) instead of 16.
  - No [CLS] token: we use Global Average Pooling over patch tokens.
  - The encoder is fully frozen (self-supervised, no learned projection).
  - ViT-Large hidden_size = 1024 (same output dim as VideoMAE-Large).

Usage:
    python scripts/vjepa2_extract_feature.py \
        --anno_root ./preprocess/Phoenix14T \
        --video_root ./phoenix_dl/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ \
        --save_dir ./datasets_features \
        --model_name facebook/vjepa2-vitl-fpc64-256 \
        --overlap_size 48 \
        --batch_size 4 \
        --device cpu
"""

import os
import os.path as osp
import argparse
import glob

import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

import sys
sys.path.append('./')

from utils.helpers import sliding_window_for_list, read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# Default frames-per-clip for V-JEPA 2 / 2.1 models
DEFAULT_FPC = 64


class VJEPA2FeatureReader(object):
    """
    Reads video frames and extracts features using a frozen V-JEPA 2.1 encoder.

    The encoder outputs patch-level tokens (no CLS token). We apply Global Average
    Pooling across the spatial-temporal patch dimension to obtain a single vector
    per clip, matching the behaviour of VideoMAE's CLS token extraction.
    """

    def __init__(
        self,
        model_name='facebook/vjepa2-vitl-fpc64-256',
        cache_dir=None,
        device='cpu',
        overlap_size=48,
        nth_layer=-1,
        frames_per_clip=DEFAULT_FPC,
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer
        self.frames_per_clip = frames_per_clip

        print(f"Loading V-JEPA 2.1 model: {model_name}")
        self.processor = AutoVideoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        ).to(self.device).eval()

        # Infer hidden dimension from config
        self.hidden_size = self.model.config.hidden_size
        print(f"  hidden_size={self.hidden_size}, frames_per_clip={self.frames_per_clip}")

    @torch.no_grad()
    def _encode_clip(self, clip_frames):
        """
        Encode a single clip of PIL Images into a feature vector.

        Args:
            clip_frames: list of PIL.Image (length = frames_per_clip)

        Returns:
            np.ndarray of shape (hidden_size,)
        """
        # Convert PIL images to numpy arrays (T, H, W, C)
        np_frames = [np.array(f) for f in clip_frames]

        inputs = self.processor(videos=[np_frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, num_patches, hidden_dim)

        # Select requested layer
        features = hidden_states[self.nth_layer]  # (1, num_patches, hidden_dim)

        # Global Average Pooling over patches (V-JEPA 2 has no CLS token)
        pooled = features.mean(dim=1)  # (1, hidden_dim)

        return pooled.cpu().numpy()

    def get_feats(self, video_frames):
        """
        Extract features from a full video using sliding window of frames_per_clip.

        Args:
            video_frames: list of PIL.Image (all frames of the video)

        Returns:
            np.ndarray of shape (num_chunks, hidden_size)
        """
        fpc = self.frames_per_clip

        # Pad if video is shorter than one clip
        if len(video_frames) < fpc:
            video_frames = list(video_frames) + [video_frames[-1]] * (fpc - len(video_frames))

        # Sliding window over frames
        clips = sliding_window_for_list(
            video_frames, window_size=fpc, overlap_size=self.overlap_size
        )

        # If sliding window produced nothing (e.g. exact length), use the full video as one clip
        if len(clips) == 0:
            clips = [video_frames[:fpc]]

        all_feats = []
        for clip in clips:
            feat = self._encode_clip(clip)  # (1, hidden_dim)
            all_feats.append(feat)

        return np.concatenate(all_feats, axis=0)  # (num_chunks, hidden_dim)

    def get_feats_batched(self, video_frames, batch_size=4):
        """
        Batched version of get_feats for better GPU throughput.

        Args:
            video_frames: list of PIL.Image
            batch_size: number of clips to process in parallel

        Returns:
            np.ndarray of shape (num_chunks, hidden_size)
        """
        fpc = self.frames_per_clip

        if len(video_frames) < fpc:
            video_frames = list(video_frames) + [video_frames[-1]] * (fpc - len(video_frames))

        clips = sliding_window_for_list(
            video_frames, window_size=fpc, overlap_size=self.overlap_size
        )

        if len(clips) == 0:
            clips = [video_frames[:fpc]]

        all_feats = []
        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i:i + batch_size]

            # Prepare batch: list of numpy video arrays
            np_batch = [[np.array(f) for f in clip] for clip in batch_clips]

            inputs = self.processor(videos=np_batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                features = outputs.hidden_states[self.nth_layer]  # (B, patches, hidden)
                pooled = features.mean(dim=1)  # (B, hidden)

            all_feats.append(pooled.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA 2.1 video features for SpaMo"
    )
    parser.add_argument('--anno_root', required=True,
                        help='Path to annotation dir (e.g. ./preprocess/Phoenix14T)')
    parser.add_argument('--video_root', required=True,
                        help='Path to video/frame root directory')
    parser.add_argument('--save_dir', required=True,
                        help='Directory to save extracted features')
    parser.add_argument('--model_name', default='facebook/vjepa2-vitl-fpc64-256',
                        help='HuggingFace model ID for V-JEPA 2.1')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of clips to process in parallel')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda:0)')
    parser.add_argument('--overlap_size', type=int, default=48,
                        help='Temporal overlap between clips (in frames)')
    parser.add_argument('--frames_per_clip', type=int, default=DEFAULT_FPC,
                        help='Number of frames per clip (default: 64)')
    parser.add_argument('--nth_layer', type=int, default=-1,
                        help='Which hidden layer to extract (-1 = last)')
    parser.add_argument('--cache_dir', default=None,
                        help='Cache directory for model weights')
    parser.add_argument('--mode', nargs='+', default=['dev', 'test', 'train'],
                        help='Dataset splits to process')
    return parser


def get_iterator(args, mode, reader):
    batch_size = args.batch_size

    data = np.load(
        os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True
    ).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    def iterate():
        for i in range(num):
            fname = data[i]['folder']
            file_id = data[i]['fileid']

            if ds_name in ('Phoenix14T', 'CSL-Daily'):
                image_list = get_img_list(ds_name, args.video_root, fname)

                if len(image_list) == 0:
                    yield np.empty((0, reader.hidden_size), dtype=np.float32), file_id, None
                    continue

                videos = [Image.open(img).convert('RGB') for img in image_list]
                feats = reader.get_feats_batched(videos, batch_size=batch_size)
                yield feats, file_id, None

            elif ds_name == 'How2Sign':
                start_time = data[i]['original_info']['START_REALIGNED']
                end_time = data[i]['original_info']['END_REALIGNED']
                videos = read_video(fname, start_time=start_time, end_time=end_time)

                if len(videos) > 0:
                    feats = reader.get_feats_batched(videos, batch_size=batch_size)
                    yield feats, file_id, str(start_time)
                else:
                    yield np.empty((0, reader.hidden_size), dtype=np.float32), file_id, str(start_time)

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    ds_name = osp.split(args.anno_root)[-1]
    _model_short = os.path.split(args.model_name)[-1]
    folder_name = f'{_model_short}_feat_{ds_name}'

    reader = VJEPA2FeatureReader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
        frames_per_clip=args.frames_per_clip,
    )

    for m in args.mode:
        print(f"\n--- Extracting split: {m} ({ds_name}) ---")
        save_path = osp.join(args.save_dir, folder_name, m)
        os.makedirs(save_path, exist_ok=True)

        # Map split names per dataset convention
        if ds_name == 'How2Sign':
            _m = 'val' if m == 'dev' else m
        elif ds_name == 'NIASL2021':
            _m = 'validation' if m == 'dev' else m
        else:
            _m = m

        generator, total_samples = get_iterator(args, _m, reader)

        for feats, file_id, st in tqdm.tqdm(generator(), total=total_samples, desc=f"Split {m}"):
            postfix = f'_overlap-{args.overlap_size}-vjepa2'
            if st is not None:
                postfix = f'_{st}{postfix}'

            out_file = osp.join(save_path, f'{file_id}{postfix}.npy')
            np.save(out_file, feats)

    print(f"\nExtraction complete! Features saved to: {osp.join(args.save_dir, folder_name)}")


if __name__ == "__main__":
    main()

