import numpy as np
from scipy.ndimage import gaussian_filter1d
from preprocess import SpikePreprocessor
from dataset import SpikeDataset
import tensorflow as tf

# ==== 加载一个样本 ====
BASE = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords"
date = "t12.2022.04.28"
path = f"{BASE}/{date}/train/chunk_0.tfrecord"

ds = SpikeDataset(
    tfrecord_paths=[path],
    compute_norm=True,
    white_noise=0.0,
    static_gain=0.0,
    smooth_sigma=2,
    stack_k=5,
    stack_stride=2,
    subsample_factor=2
)

# 原始取一条
example = ds.samples[0]
feature_description = ds.feature_description
e = tf.io.parse_single_example(example, feature_description)

X_raw = e["inputFeatures"].numpy().astype(np.float32)  # (T,256)
T_raw = X_raw.shape[0]
print("===================================================")
print("原始数据:")
print("T_raw =", T_raw)
print("shape =", X_raw.shape)
print("前 3 帧:")
print(X_raw[:3])
print("===================================================")

# ==== step 1: normalize ====
mean, std = ds.session_stats[path]
X_norm = (X_raw - mean) / std
print("\nAfter Normalize:")
print("shape =", X_norm.shape)
print("前 3 帧:")
print(X_norm[:3])

# ==== step 2: smoothing ====
X_smooth = gaussian_filter1d(X_norm, sigma=2, axis=0)
print("\nAfter Smoothing:")
print("shape =", X_smooth.shape)
print("前 3 帧:")
print(X_smooth[:3])

# ==== step 3: stack ====
stack_k = 5
stack_stride = 2

T_stack = (T_raw - stack_k) // stack_stride + 1
X_stack = np.zeros((T_stack, 256 * stack_k), dtype=np.float32)

idx = 0
for t in range(0, T_raw - stack_k + 1, stack_stride):
    patch = X_smooth[t:t+stack_k].reshape(-1)
    X_stack[idx] = patch
    idx += 1

print("\nAfter Time Stack:")
print("T_stack =", T_stack)
print("shape =", X_stack.shape)
print("前 3 帧 feature 长度 =", X_stack[0].shape[0])
print("前 3 帧:")
print(X_stack[:3])

# ==== step 4: subsample ====
subsample_factor = 2
X_sub = X_stack[::subsample_factor]
T_sub = X_sub.shape[0]

print("\nAfter Subsample:")
print("subsample_factor =", subsample_factor)
print("T_sub =", T_sub)
print("shape =", X_sub.shape)
print("前 3 帧:")
print(X_sub[:3])

# ==== step 5: 最终结果（你的preprocessor的结果） ====
pre = SpikePreprocessor(
    smooth_sigma=2,
    stack_k=5,
    stack_stride=2,
    subsample_factor=2
)
X_final, T_final = pre(X_raw)

print("\n===================================================")
print("最终预处理结果（通过preprocessor）：")
print("T_final =", T_final)
print("shape =", X_final.shape)
print("前 3 帧:")
print(X_final[:3])
print("===================================================")
