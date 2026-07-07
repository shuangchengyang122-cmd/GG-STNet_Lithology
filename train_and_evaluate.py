import os
import random
import csv
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 模型文件自行保证同目录存在
from gg_stnet_model import build_gg_stnet

# 屏蔽TF冗余日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

# -----------------------------
# Config (paper aligned)
# -----------------------------
FEATURES = ["GR", "AC", "DEN", "RT", "SP"]
LABEL_COL = "Lithology_Label"

NUM_CLASSES = 7
WINDOW_SIZE = 32
STEP_SIZE = 16
LABEL_MODE = "center"

BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 20

MODEL_DIR = Path("outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 固定随机种子，实验可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Data utilities
# -----------------------------
def read_split_file(txt_path):
    if not Path(txt_path).exists():
        raise FileNotFoundError(f"划分文件不存在：{txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]
    valid_fp = []
    for p in lines:
        p_path = Path(p)
        if p_path.exists():
            valid_fp.append(str(p_path))
        else:
            print(f"警告：井文件丢失 {p}，自动跳过")
    if len(valid_fp) == 0:
        raise ValueError(f"{txt_path} 无有效井数据")
    return valid_fp


def load_single_well(csv_path):
    df = pd.read_csv(csv_path)
    # 校验特征列与标签列
    miss_feat = [f for f in FEATURES if f not in df.columns]
    if len(miss_feat) > 0:
        raise ValueError(f"{csv_path} 缺失测井曲线：{miss_feat}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"{csv_path} 缺失标签列 {LABEL_COL}")

    x = df[FEATURES].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)

    # 标签空值校验
    if np.any(pd.isna(y)):
        raise ValueError(f"{csv_path} has NaN labels")
    # 标签范围校验
    if y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError(f"{csv_path} 标签超出合法范围 0~{NUM_CLASSES-1}")
    # 测井曲线空值填充
    if np.any(np.isnan(x)):
        print(f"警告 {csv_path} 测井曲线存在空值，0填充")
        x = np.nan_to_num(x, nan=0.0)
    return x, y


# -----------------------------
# Scaler (paper-consistent)
# -----------------------------
def fit_scaler(train_files):
    all_x = []
    for fp in train_files:
        x, _ = load_single_well(fp)
        all_x.append(x)
    all_x = np.concatenate(all_x, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_x)
    return scaler


# -----------------------------
# Window generation（修复末尾数据丢弃，反射填充）
# -----------------------------
def make_windows(x, y, scaler):
    x = scaler.transform(x).astype(np.float32)
    windows, labels = [], []
    offset = WINDOW_SIZE // 2
    seq_len = len(x)

    for i in range(0, seq_len, STEP_SIZE):
        end = i + WINDOW_SIZE
        if end > seq_len:
            pad_len = end - seq_len
            win = np.pad(x[i:], ((0, pad_len), (0, 0)), mode="reflect")
        else:
            win = x[i:end]
        center = i + offset
        if center >= len(y):
            continue
        windows.append(win)
        labels.append(y[center])
    return np.array(windows), np.array(labels)


# -----------------------------
# TF Dataset 流式加载，避免内存爆炸
# -----------------------------
def well_sample_generator(files, scaler):
    while True:
        for fp in files:
            x, y = load_single_well(fp)
            wx, wy = make_windows(x, y, scaler)
            for xi, yi in zip(wx, wy):
                yield xi, yi


def get_tf_dataset(file_list, scaler, batch_size, shuffle=True):
    gen = lambda: well_sample_generator(file_list, scaler)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_SIZE, len(FEATURES)), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Standard block-level evaluation
# -----------------------------
def evaluate_split(model, dataset, name):
    loss, acc = model.evaluate(dataset, verbose=0)
    # 全量预测
    all_x, all_y = [], []
    for batch_x, batch_y in dataset.unbatch().batch(1024):
        all_x.append(batch_x.numpy())
        all_y.append(batch_y.numpy())
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    pred = np.argmax(model.predict(x_all, verbose=0, batch_size=512), axis=1)

    print(f"\n[{name}] Loss={loss:.4f} Acc={acc:.4f}")
    print(classification_report(y_all, pred, digits=4, zero_division=0))
    print(confusion_matrix(y_all, pred))
    return pred, acc


# =========================================================
# 🔥 GEOLOGICAL CONSISTENCY METRICS (TABLE 6)
# =========================================================

def reconstruct_well_prediction(model, x, y, scaler):
    """修复：批量预测 + 边界缺失预测填充"""
    x = scaler.transform(x).astype(np.float32)
    seq_len = len(y)
    pred_sum = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
    count_sum = np.zeros(seq_len, dtype=np.int32)
    offset = WINDOW_SIZE // 2

    window_batch = []
    center_list = []
    max_start = len(x) - WINDOW_SIZE
    for i in range(0, max_start + 1, STEP_SIZE):
        window = x[i:i + WINDOW_SIZE]
        window_batch.append(window)
        center_list.append(i + offset)

    # 批量预测加速
    if len(window_batch) > 0:
        batch_prob = model.predict(np.array(window_batch), verbose=0, batch_size=256)
        for prob, center in zip(batch_prob, center_list):
            if center < seq_len:
                pred_sum[center] += prob
                count_sum[center] += 1

    seq_pred = np.zeros(seq_len, dtype=int)
    mask_valid = count_sum > 0
    seq_pred[mask_valid] = np.argmax(pred_sum[mask_valid], axis=-1)

    # 前向填充边界空白
    last_valid = None
    for idx in range(seq_len):
        if mask_valid[idx]:
            last_valid = seq_pred[idx]
        elif last_valid is not None:
            seq_pred[idx] = last_valid
    # 后向兜底
    last_valid = None
    for idx in reversed(range(seq_len)):
        if mask_valid[idx]:
            last_valid = seq_pred[idx]
        elif last_valid is not None:
            seq_pred[idx] = last_valid
    return seq_pred.astype(int)


# -----------------------------
# 1. boundary offset error
# -----------------------------
def boundary_offset(y_true, y_pred, step_m=0.125):
    true_b = np.where(np.diff(y_true) != 0)[0]
    pred_b = np.where(np.diff(y_pred) != 0)[0]
    if len(true_b) == 0 or len(pred_b) == 0:
        return np.nan
    dist = []
    for t in true_b:
        dist.append(np.min(np.abs(pred_b - t)))
    return np.mean(dist) * step_m


# -----------------------------
# 2. thin bed recognition (修复分段匹配逻辑)
# -----------------------------
def thin_bed_recognition(y_true, y_pred, threshold=3):
    def get_seg_info(y_seq):
        seg_list = []
        start = 0
        for i in range(1, len(y_seq)):
            if y_seq[i] != y_seq[i-1]:
                seg_list.append({
                    "label": y_seq[start],
                    "start": start,
                    "end": i - 1,
                    "len": i - start
                })
                start = i
        seg_list.append({
            "label": y_seq[start],
            "start": start,
            "end": len(y_seq) - 1,
            "len": len(y_seq) - start
        })
        return seg_list

    true_segs = get_seg_info(y_true)
    pred_segs = get_seg_info(y_pred)
    thin_total = 0
    thin_correct = 0

    for ts in true_segs:
        if ts["len"] > threshold:
            continue
        thin_total += 1
        overlap_correct = False
        for ps in pred_segs:
            if not (ps["end"] < ts["start"] or ps["start"] > ts["end"]):
                if ps["label"] == ts["label"]:
                    overlap_correct = True
                    break
        if overlap_correct:
            thin_correct += 1
    if thin_total == 0:
        return np.nan
    return thin_correct / thin_total


# -----------------------------
# 3. isolated segment rate
# -----------------------------
def isolated_rate(y_pred, max_len=2):
    segs = []
    start = 0
    for i in range(1, len(y_pred)):
        if y_pred[i] != y_pred[i-1]:
            segs.append(i - start)
            start = i
    segs.append(len(y_pred) - start)
    return sum(s <= max_len for s in segs) / (len(segs) + 1e-8)


# -----------------------------
# full well-level evaluation (TABLE 5 + 6)
# -----------------------------
def evaluate_wells(model, files, scaler, name):
    accs, f1s = [], []
    b_offs, thin_rs, iso_rs = [], [], []

    for fp in files:
        x, y = load_single_well(fp)
        wx, wy = make_windows(x, y, scaler)
        pred = np.argmax(model.predict(wx, verbose=0, batch_size=512), axis=1)
        accs.append(np.mean(pred == wy))
        rep = classification_report(wy, pred, output_dict=True, zero_division=0)
        f1s.append(rep["macro avg"]["f1-score"])

        seq_pred = reconstruct_well_prediction(model, x, y, scaler)
        b_offs.append(boundary_offset(y, seq_pred))
        thin_rs.append(thin_bed_recognition(y, seq_pred))
        iso_rs.append(isolated_rate(seq_pred))

    print(f"\n[{name} WELL-LEVEL RESULTS]")
    print(f"Accuracy mean: {np.mean(accs):.4f}")
    print(f"Accuracy std : {np.std(accs):.4f}")
    print(f"Macro-F1 mean: {np.mean(f1s):.4f}")

    print("\n[Geological Metrics]")
    bo_mean = np.nanmean(b_offs)
    tb_mean = np.nanmean(thin_rs)
    ir_mean = np.nanmean(iso_rs)
    if np.isnan(bo_mean):
        print("警告：边界偏移指标全部为空（无地层分界面）")
    if np.isnan(tb_mean):
        print("警告：数据集无薄层样本")
    print(f"Boundary-offset (m): {bo_mean:.4f}")
    print(f"Thin-bed recognition: {tb_mean:.4f}")
    print(f"Isolated rate: {ir_mean:.4f}")
    return np.mean(accs), np.mean(f1s), bo_mean, tb_mean, ir_mean


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    train_files = read_split_file(r"E:\splits\train_wells.txt")
    val_files = read_split_file(r"E:\splits/val_wells.txt")
    test_files = read_split_file(r"E:\splits/test_wells.txt")
    blind_files = read_split_file(r"E:\splits/blind_wells.txt")

    print("[1] 拟合训练集标准化器...")
    scaler = fit_scaler(train_files)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    print("[2] 构建TF流式数据集...")
    train_ds = get_tf_dataset(train_files, scaler, BATCH_SIZE, shuffle=True)
    val_ds = get_tf_dataset(val_files, scaler, BATCH_SIZE, shuffle=False)
    test_ds = get_tf_dataset(test_files, scaler, BATCH_SIZE, shuffle=False)

    print("[3] 初始化GG-STNet模型...")
    model = build_gg_stnet(WINDOW_SIZE, len(FEATURES), NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best.keras"),
            save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "last_epoch.keras"),
            save_best_only=False
        ),
        tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "train_log.csv"))
    ]

    print("[4] 开始训练...")
    # 估算steps，可根据实际总样本微调
    steps_per_epoch = len(train_files) * 800
    val_steps = len(val_files) * 300
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    print("[5] Test集分块指标评估")
    _, test_acc = evaluate_split(model, test_ds, "TEST")

    print("[6] 盲井井级地质指标评估")
    blind_acc, blind_f1, bo_m, thin_rec, iso_r = evaluate_wells(model, blind_files, scaler, "BLIND")

    # 保存全部实验指标到csv
    res_path = MODEL_DIR / "eval_summary.csv"
    with open(res_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_size", "step_size", "lr", "batch_size",
            "test_block_acc", "blind_well_mean_acc", "blind_macro_f1",
            "boundary_offset_m", "thin_recall", "isolated_segment_rate"
        ])
        writer.writerow([
            WINDOW_SIZE, STEP_SIZE, LEARNING_RATE, BATCH_SIZE,
            round(test_acc,4), round(blind_acc,4), round(blind_f1,4),
            round(bo_m,4), round(thin_rec,4), round(iso_r,4)
        ])
    print(f"\n实验指标已保存至：{res_path.resolve()}")
    print("[训练&评估流程全部完成]")
