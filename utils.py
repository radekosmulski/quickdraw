import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, BatchSampler

# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892
BASE_SIZE = 256
def list2drawing(raw_strokes, size=256, lw=6, time_color=False):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img

def drawing2tensor(drawing):
    rgb = cv2.cvtColor(drawing,cv2.COLOR_GRAY2RGB)
    rgb = rgb.transpose(2,0,1).astype(np.float32)
    return torch.from_numpy(rgb)

def channels2tensor(ch1, ch2, ch3):
    rgb = np.stack((ch1, ch2, ch3)).astype(np.float32)
    return torch.from_numpy(rgb)

# https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw
def drawing2seq(inkarray):
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    if not inkarray:
        print("Empty inkarray")
        return None, None
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None, None
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1    # stroke_end
    # Preprocessing.
    # 1. Size normalization.
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]
    return np_ink

# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def map3(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_3 = predicted_idxs[:, :3]
    res = mapk([[t] for t in targs.cpu().numpy()], top_3.cpu().numpy(), 3)
    return torch.tensor(res)

def top_3_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :3]

def top_3_pred_labels(preds, classes):
    top_3 = top_3_preds(preds)
    labels = []
    for i in range(top_3.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_3[i]]))
    return labels

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class RandomSamplerWithEpochSize(Sampler):
    """Yields epochs of specified sizes. Iterates over all examples in a data_source in random
    order. Ensures (nearly) all examples have been trained on before beginning the next iteration
    over the data_source - drops the last epoch that would likely be smaller than epoch_size.
    """
    def __init__(self, data_source, epoch_size):
        self.n = len(data_source)
        self.epoch_size = epoch_size
        self._epochs = []
    def __iter__(self):
        return iter(self.next_epoch)
    @property
    def next_epoch(self):
        if len(self._epochs) == 0: self.generate_epochs()
        return self._epochs.pop()
    def generate_epochs(self):
        idxs = [i for i in range(self.n)]
        np.random.shuffle(idxs)
        self._epochs = list(chunks(idxs, self.epoch_size))[:-1]
    def __len__(self):
        return self.epoch_size

def create_submission(test_preds, test_dl, name, classes):
    key_ids = [path.stem for path in test_dl.dataset.x.items]
    labels = top_3_pred_labels(test_preds, classes)
    sub = pd.DataFrame({'key_id': key_ids, 'word': labels})
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')
