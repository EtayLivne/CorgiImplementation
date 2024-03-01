import io
import os
from typing import List

import boto3
import numpy as np

from constants import ENDPOINT_URL, NPZ_COLS, IMG_COLS
import torch
import torchvision
from torchvision.transforms import functional as F

LM2_HEIGHT = 960
LM2_WIDTH = 1260
CROP_HEIGHT = 240
CROP_WIDTH = 320
FOCUS_M = 40

def create_crops(data, features_names, paths_names):
    cropped_features = []
    cropped_paths = []

    for feature_name, path_name in zip(features_names, paths_names, strict=False):
        cam = feature_name.split('_')[0]
        timestamp = '' if '_t-' not in feature_name else '_t-2'
        m_proj = data[f'{cam}_m_projection{timestamp}'][..., 0]
        points = m_proj[(m_proj > FOCUS_M) | (m_proj < -FOCUS_M)]

        if len(points) > 0:
            coord = np.where(m_proj == points[np.argmin(np.abs(points))])
        else:
            coord = ([240], [640])  # An arbitrary central location in LM2.

        img = torch.from_numpy(data[feature_name][0])
        crop = F.crop(
            img,
            top=min(max(0, coord[0][0] - CROP_HEIGHT // 2), LM2_HEIGHT - CROP_HEIGHT),
            left=min(max(0, coord[1][0] - CROP_WIDTH // 2), LM2_WIDTH - CROP_WIDTH),
            height=CROP_HEIGHT,
            width=CROP_WIDTH,
        )

        path = torch.from_numpy(data[path_name]).permute(2, 0, 1)
        crop_path = F.crop(
            path,
            top=min(max(0, coord[0][0] - CROP_HEIGHT // 2), LM2_HEIGHT - CROP_HEIGHT),
            left=min(max(0, coord[1][0] - CROP_WIDTH // 2), LM2_WIDTH - CROP_WIDTH),
            height=CROP_HEIGHT,
            width=CROP_WIDTH,
        )

        if len(points) == 0 and cam not in ['main', 'rear']:
            crop_path = torch.zeros_like(crop_path)
            crop = torch.zeros_like(crop)

        cropped_features.append(crop)
        cropped_paths.append(crop_path.permute(1, 2, 0))

    cropped_features = torch.stack(cropped_features, dim=0).unsqueeze(-1)
    cropped_paths = torch.stack(cropped_paths, dim=0)

    return cropped_features, cropped_paths


def get_merge_camera_inputs(cameras=None, num_crops=None, multi_frame=True):
    if cameras is None:
        cameras = ['main']
    
    cameras = [cam.lower() for cam in cameras]
    
    timesteps = ['']
    if multi_frame:
        timesteps += ['_t-2']

    features_names = [f'{cam}_features{t}' for t in timesteps for cam in cameras]
    paths_names = [f'{cam}_paths{t}' for t in timesteps for cam in cameras]

    def merge_camera_inputs(data):
        merged_features = np.concatenate([data[key] for key in features_names if 'features' in key], axis=0)[..., None]
        merged_paths = np.stack([data[key] for key in paths_names], axis=0)

        if num_crops > 1:
            cropped_features, cropped_paths = create_crops(data, features_names, paths_names)
            merged_features = F.resize(
                torch.from_numpy(merged_features[..., 0]), (CROP_HEIGHT, CROP_WIDTH), antialias=True
            ).unsqueeze(-1)
            DOWNSAMPLE_FACTOR = 4
            all_paths = torch.from_numpy(
                merged_paths.reshape(-1, CROP_HEIGHT, DOWNSAMPLE_FACTOR, CROP_WIDTH, DOWNSAMPLE_FACTOR, 3).max(4).max(2)
            )
            all_features = torch.concat([merged_features, cropped_features], dim=0)
            all_paths = torch.concat([all_paths, cropped_paths], dim=0)
        else:
            all_features = merged_features
            all_paths = merged_paths

        data['all_features'] = all_features
        data['all_paths'] = all_paths

        return data

    return merge_camera_inputs


def get_preprocess_filter_objects(x_bound=10, z_bound=40):
    """Creates a preprocess function that filters objects for the chosen path.
    The filtering is represented in the data['objects_mask'] mask.
    """

    def preprocess_filter_objects(data):
        objects = data['boxes'].reshape(100, 30)
        objects = objects[~np.all(objects == -999, axis=-1)]
        objects = objects[:, [2,0,7,4,5]]
        data['objects'] = objects

        objects_not_far_mask = (np.abs(objects[:, 1]) <= x_bound) & (np.abs(objects[:, 0]) <= z_bound)
        data['objects_mask'] = objects_not_far_mask

        return data

    return preprocess_filter_objects


def get_preprocess_tokenize_objects(tokenizer, shuffle_objects=True):
    """Creates a preprocess function that use the given tokenizer to turn the objects
    into a sequnce.
    """

    def preprocess_tokenize_objects(data):
        objects = data['objects']
        future_objects = data['objects']
        del data['objects']
        objects_mask = data['objects_mask']
        del data['objects_mask']
        objects_for_tokenizer = objects[objects_mask]
        ZF_IDX = 0
        XF_IDX = 1
        future_objects_z_x = future_objects[objects_mask][:, [ZF_IDX, XF_IDX]]
        future_objects_z_x = np.nan_to_num(future_objects_z_x)
        # TODO: do we want to move tokenizer to numpy?
        full_objects = np.concatenate([objects_for_tokenizer, future_objects_z_x], -1)
        sequence, weights = tokenizer(torch.tensor(full_objects), shuffle=shuffle_objects)
        sequence = sequence.numpy().astype(np.int64)
        data['input_sequence'] = sequence
        padded_tokens = np.pad(sequence[1:], [[0, 1]], constant_values=tokenizer.PAD_code)
        data['output_sequence'] = padded_tokens
        return data

    return preprocess_tokenize_objects


def get_preprocess_concat_paths(cameras=None, multi_frame=True):
    """Creates a preprocess function that use the given tokenizer to turn the objects
    into a sequnce.
    """
    if cameras is None:
        cameras = ['main']

    cameras = [cam.lower() for cam in cameras]
    def preprocess_concat_paths(data):
        for cam in cameras:
            x_features = data[f'{cam}_x']
            z_features = data[f'{cam}_z']
            semantic_features = data[f'{cam}_semantic']
            data[f'{cam}_paths'] = np.stack([x_features, z_features, semantic_features], axis=-1).astype(np.float32)
            if multi_frame:
                x_features = data[f'{cam}_x_t-2']
                z_features = data[f'{cam}_z_t-2']
                semantic_features = data[f'{cam}_semantic_t-2']
                data[f'{cam}_paths_t-2'] = np.stack([x_features, z_features, semantic_features], axis=-1).astype(np.float32)
        return data

    return preprocess_concat_paths


def read_jpeg(path: str, scale: bool = True):
    s3 = boto3.resource("s3", 
                        endpoint_url=ENDPOINT_URL,
                        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    jpeg_path = path.replace("s3://", "")
    sep = jpeg_path.find("/")
    bucket_name = jpeg_path[:sep]
    jpeg_key = jpeg_path[sep + 1 :]
    frame_id = jpeg_key.split("/")[-1].split(".")[0].split('-')[-1]
    
    try:
        response = s3.Bucket(bucket_name).Object(jpeg_key).get()
        image_bytes = response["Body"].read()
        torch_bytes = torch.tensor(list(image_bytes), dtype=torch.uint8)
        tensor_image = torchvision.io.decode_jpeg(torch_bytes)
    except Exception as e:
        print('error in t frame key:')
        print(jpeg_key)
        print(e)
        tensor_image = 0.0
        
    try:
        response_past = s3.Bucket(bucket_name).Object(jpeg_key.replace(f'-{frame_id}.', f'-{str(int(frame_id)-4)}.')).get()
        image_bytes_past = response_past["Body"].read()
        torch_bytes_past = torch.tensor(list(image_bytes_past), dtype=torch.uint8)
        tensor_image_past = torchvision.io.decode_jpeg(torch_bytes_past)
    except Exception as e:
        print('error in t-2 frame key:')
        print(jpeg_key.replace(f'-{frame_id}.', f'-{str(int(frame_id)-4)}.'))
        print(e)
        tensor_image_past = 0.0

    if scale:
        tensor_image = tensor_image / 255.0
        tensor_image_past = tensor_image_past / 255.0

    return tensor_image, tensor_image_past


def read_jpeg_columns(row_dict: dict, img_cols: List[str] = IMG_COLS, scale: bool = True):
    for col in img_cols:
        row_dict[col], row_dict[f"{col}_t-2"] = read_jpeg(row_dict[col], scale)
    return row_dict


def decompress_numpy_arrays(compressed_data):
    buf = io.BytesIO(compressed_data)
    data = np.load(buf)
    nd_array = data['arr']
    return nd_array


def read_npz_columns(row_dict: dict, npz_cols: List[str] = NPZ_COLS):
    for col in npz_cols:
        row_dict[col.replace("_t_2", "_t-2")] = decompress_numpy_arrays(row_dict[col])
    return row_dict
