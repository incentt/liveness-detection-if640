"""
Active Liveness Detection — e-KYC Demo
run: /opt/anaconda3/envs/liveness/bin/python liveness_demo.py

Webcam controls:
  Q / Esc  = keluar
  S        = mulai sesi baru (challenge diacak ulang)
"""

import matplotlib
matplotlib.use('Agg')

import cv2, numpy as np, mediapipe as mp
import matplotlib.pyplot as plt, seaborn as sns
import os, random, time, uuid, hashlib, math
from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial import distance as dist
from tqdm import tqdm
from PIL import Image

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                              accuracy_score, f1_score)

import face_recognition

import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T, torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import warnings; warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = (torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))

MODEL_PATH = Path('liveness_cnn_best.pth')
DATA_ROOT  = Path('./data/CelebA_Spoof')
DATA_DIR   = DATA_ROOT / 'Data'
IMG_SIZE   = 224   # upgraded from 112 — better texture discrimination

# ═══════════════════════════════════════════════════════════
# 1. MEDIAPIPE + FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════

mp_face_mesh  = mp.solutions.face_mesh
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
HEAD_POSE_IDX = [1, 152, 226, 446, 57, 287]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)


def compute_EAR(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-7)


def detect_blink(ear_history, threshold=0.21, open_threshold=0.26, consec_frames=2):
    """
    Requires a full open → close → open cycle.
    A static photo with constant low EAR will NOT pass because EAR never
    transitions from open→closed; it has no prior open state.
    """
    arr = list(ear_history)
    if len(arr) < consec_frames + 2:
        return False, 0
    count = 0
    i = 1
    while i < len(arr):
        # Must start from an open-eye state
        if arr[i - 1] >= open_threshold:
            # Count consecutive closed frames
            j = i
            while j < len(arr) and arr[j] < threshold:
                j += 1
            n_closed = j - i
            if n_closed >= consec_frames:
                # Eye must reopen after the close
                reopened = any(arr[k] >= open_threshold * 0.88 for k in range(j, min(j + 8, len(arr))))
                if reopened:
                    count += 1
                    i = j + 1
                    continue
        i += 1
    return count > 0, count


def get_head_pose(landmarks, w, h):
    img_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in HEAD_POSE_IDX],
        dtype=np.float64)
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, img_pts, cam,
                                np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    ang, *_ = cv2.RQDecomp3x3(rmat)
    return ang[1], ang[0], ang[2]


def compute_face_size(landmarks, w, h):
    """
    Face height (forehead → chin) normalized by frame height.
    More stable than IOD when head is turned sideways.
    """
    forehead_y = landmarks[10].y * h   # top of forehead
    chin_y     = landmarks[152].y * h  # chin
    return abs(chin_y - forehead_y) / h


def compute_face_center(landmarks):
    """Returns (cx, cy) normalized face center in [0, 1]."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return float(np.mean(xs)), float(np.mean(ys))


def compute_vertical_ratio(landmarks, h):
    """
    (chin_y - nose_y) / (chin_y - eye_y).
    ~0.50 neutral, decreases when looking up (chin rises), increases when looking down.
    More stable than solvePnP for small upward tilts.
    """
    nose_y = landmarks[4].y * h
    eye_y  = (landmarks[33].y + landmarks[263].y) / 2.0 * h
    chin_y = landmarks[152].y * h
    denom  = abs(chin_y - eye_y)
    return (chin_y - nose_y) / denom if denom > 1.0 else 0.50


def extract_face_encoding(frame_bgr):
    """
    Returns 128-dim face encoding using face_recognition (dlib HOG model).
    Returns None if no face found. Fast enough to call every ~20 frames.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model='hog')
    if not locs:
        return None
    encs = face_recognition.face_encodings(rgb, locs)
    return encs[0] if encs else None


def compute_motion_score(nose_buf, ear_buf):
    """
    Micro-motion liveness score (0..1).
    Real faces have natural jitter from breathing/micro-movements (~0.3+).
    Still photos / printed images score near 0 (~0.03-0.08).
    Uses nose-tip positional variance + EAR temporal variance.
    """
    score = 0.0
    if len(nose_buf) >= 15:
        pts     = np.array(list(nose_buf))
        pos_std = float(np.std(pts[:, 0]) + np.std(pts[:, 1]))
        score  += min(1.0, pos_std / 0.003)   # 0.003 normalized units ≈ real face jitter
    if len(ear_buf) >= 15:
        ear_std = float(np.std(list(ear_buf)))
        score  += min(1.0, ear_std / 0.006)   # 0.006 EAR std ≈ real eye micro-motion
    return min(1.0, score / 2.0)


def _smooth_series(series, n=3):
    """Rolling mean of window n applied across a list."""
    arr = list(series)
    return [float(np.mean(arr[max(0, i - n + 1):i + 1])) for i in range(len(arr))]


def lbp_histogram(gray, bins=64):
    img = cv2.resize(gray, (64, 64))
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    h, _ = np.histogram(mag.ravel(), bins=bins, range=(0, 300))
    h = h.astype(np.float32)
    return h / (h.sum() + 1e-7)


# ═══════════════════════════════════════════════════════════
# 2. CNN MODEL
# ═══════════════════════════════════════════════════════════

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

VAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    # Colour/brightness shifts — simulate phone screen vs. real skin
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.08),
    T.RandomRotation(15),
    # Blur — simulates phone screen softness / printed photo softness
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))], p=0.45),
    # Slight perspective warp — photo held at angle
    T.RandomPerspective(distortion_scale=0.15, p=0.3),
    # Occasional greyscale — B&W printouts
    T.RandomGrayscale(p=0.06),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


class FocalBCELoss(nn.Module):
    """Binary focal loss — down-weights easy negatives so model focuses on hard spoofs."""
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha

    def forward(self, logits, targets):
        import torch.nn.functional as F
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class LivenessCNN(nn.Module):
    def __init__(self, pretrained=True, freeze_stages=True):
        super().__init__()
        bb = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
        if freeze_stages:
            for p in bb.features[:14].parameters(): p.requires_grad = False
        in_f = bb.classifier[1].in_features
        bb.classifier = nn.Sequential(
            nn.Dropout(0.4),  nn.Linear(in_f, 256), nn.ReLU(True),
            nn.Dropout(0.25), nn.Linear(256, 64),   nn.ReLU(True),
            nn.Linear(64, 1)
        )
        self.model = bb

    def unfreeze_all(self):
        for p in self.model.parameters(): p.requires_grad = True

    def forward(self, x): return self.model(x).squeeze(1)


@torch.no_grad()
def score_frame(model, frame_bgr, face_bbox=None):
    """
    face_bbox: (x1,y1,x2,y2) pixel coords of face region from MediaPipe landmarks.
    Cropping to the face before scoring dramatically improves CNN accuracy — the model
    sees skin/screen texture instead of a tiny face in a huge background.
    """
    if model is None: return 1.0
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        crop = frame_bgr[y1:y2, x1:x2]
        src  = crop if crop.size > 0 else frame_bgr
    else:
        src = frame_bgr
    img = cv2.cvtColor(cv2.resize(src, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    t   = VAL_TRANSFORM(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
    model.eval()
    return float(torch.sigmoid(model(t)).cpu().item())


# ═══════════════════════════════════════════════════════════
# 3. TRAINING PIPELINE  (dijalankan hanya jika model belum ada)
# ═══════════════════════════════════════════════════════════

def read_bb(bb_path):
    try:
        vals = open(bb_path).read().strip().split()
        return int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])
    except: return None


def crop_bb(img, bb_path, pad=0.15):
    bb = read_bb(bb_path)
    if bb is None: return img
    x, y, w, h = bb; ih, iw = img.shape[:2]
    px, py = int(w*pad), int(h*pad)
    crop = img[max(0,y-py):min(ih,y+h+py), max(0,x-px):min(iw,x+w+px)]
    return crop if crop.size > 0 else img


def load_dataset():
    samples = []
    for subj in sorted(DATA_DIR.rglob('*')):
        if not subj.is_dir(): continue
        for cat, lbl in [('live', 1), ('spoof', 0)]:
            cat_dir = subj / cat
            if not cat_dir.exists(): continue
            for p in cat_dir.glob('*.png'):
                if '_BB' not in p.name: samples.append((str(p), lbl, subj.parent.name))
    random.shuffle(samples)
    n_live = sum(1 for _,l,_ in samples if l==1)
    print(f'Dataset: {len(samples):,} gambar | live={n_live:,} | spoof={len(samples)-n_live:,}')
    return samples


def split_by_subject(samples):
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    groups = [Path(p).parent.parent.name for p in paths]
    idx    = list(range(len(samples)))
    tv_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
                          .split(idx, labels, groups))
    sub_l  = [labels[i] for i in tv_idx]
    sub_g  = [groups[i] for i in tv_idx]
    r_tr, r_val = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
                       .split(tv_idx, sub_l, sub_g))
    return ([samples[tv_idx[i]] for i in r_tr],
            [samples[tv_idx[i]] for i in r_val],
            [samples[te_idx[i]] for i in range(len(te_idx))])


_mesh_static = None

def _get_mesh():
    global _mesh_static
    if _mesh_static is None:
        _mesh_static = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    return _mesh_static


def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = crop_bb(img, img_path.replace('.png', '_BB.txt'))
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbp  = lbp_histogram(gray)
    cfeat = [c for ch in range(3) for c in [img[:,:,ch].mean()/255, img[:,:,ch].std()/255]]
    lap  = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hfeat = [hsv[:,:,c].mean()/255 for c in range(3)] + [hsv[:,:,c].std()/255 for c in range(3)]
    ear_feat = [0.28, 0.28, 0.0]
    res = _get_mesh().process(rgb)
    if res.multi_face_landmarks:
        lm    = res.multi_face_landmarks[0].landmark
        el    = compute_EAR(lm, LEFT_EYE_IDX, w, h)
        er    = compute_EAR(lm, RIGHT_EYE_IDX, w, h)
        ear_feat = [el, er, abs(el - er)]
    _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    hi = bright.sum() / (255 * h * w)
    return np.concatenate([lbp, cfeat, hfeat, [lap], ear_feat, [hi]]).astype(np.float32)


class SpoofDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = [(p, l) for p, l, _ in samples if os.path.exists(p)]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        else:
            bb = read_bb(path.replace('.png', '_BB.txt'))
            if bb:
                x,y,w,h = bb; ih,iw = img.shape[:2]; pad=0.15
                img = img[max(0,int(y-h*pad)):min(ih,int(y+h*(1+pad))),
                          max(0,int(x-w*pad)):min(iw,int(x+w*(1+pad)))]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(img)), torch.tensor(label, dtype=torch.float32)

    def class_weights(self):
        lbls = [l for _,l in self.samples]; cnt = Counter(lbls); total = len(lbls)
        return [total/(2*cnt[l]) for l in lbls]


def run_training():
    print('\n' + '='*60)
    print('MODEL BELUM ADA — Mulai training...')
    print('='*60)

    if not DATA_DIR.exists():
        print(f'ERROR: Dataset tidak ditemukan di {DATA_DIR}')
        print('Pastikan struktur: data/CelebA_Spoof/Data/test/{subj}/live/*.png')
        return None, None, None

    samples = load_dataset()
    train_s, val_s, test_s = split_by_subject(samples)
    print(f'Split: train={len(train_s):,} | val={len(val_s):,} | test={len(test_s):,}')

    # ── Feature extraction ──
    print('\n[1/3] Ekstraksi fitur...')
    def build_X(samps, desc):
        X, y = [], []
        for p, lbl, _ in tqdm(samps, desc=desc):
            f = extract_features(p)
            if f is not None: X.append(f); y.append(lbl)
        return np.array(X, np.float32), np.array(y)

    X_tr_r, y_tr = build_X(train_s, 'Train')
    X_va_r, y_va = build_X(val_s,   'Val  ')
    X_te_r, y_te = build_X(test_s,  'Test ')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_r)
    X_va = scaler.transform(X_va_r)
    X_te = scaler.transform(X_te_r)

    # ── ML classifiers ──
    print('\n[2/3] Training ML classifiers...')
    clfs = {
        'SVM':   SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED),
        'RF':    RandomForestClassifier(200, max_depth=12, random_state=SEED),
        'GBM':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED),
    }
    ml_res = {}; best_auc = 0; best_ml = None
    print(f'  {"Model":6s}  Acc    AUC   ACER')
    for name, clf in clfs.items():
        clf.fit(X_tr, y_tr)
        yp = clf.predict(X_te); yprob = clf.predict_proba(X_te)[:,1]
        auc  = roc_auc_score(y_te, yprob)
        acc  = accuracy_score(y_te, yp)
        cm   = confusion_matrix(y_te, yp); TN,FP,FN,TP = cm.ravel()
        acer = ((FP/(TN+FP+1e-7)) + (FN/(FN+TP+1e-7))) / 2
        ml_res[name] = dict(auc=auc, acc=acc, acer=acer, model=clf, yp=yp, yprob=yprob)
        print(f'  {name:6s}  {acc:.3f}  {auc:.3f}  {acer:.3f}')
        if auc > best_auc: best_auc = auc; best_ml = clf

    # ── ML plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ML Classifier Evaluation — CelebA-Spoof', fontweight='bold')
    for (name, r), c in zip(ml_res.items(), ['#e41a1c','#377eb8','#4daf4a']):
        fpr, tpr, _ = roc_curve(y_te, r['yprob'])
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} AUC={r['auc']:.3f}")
    axes[0].plot([0,1],[0,1],'k--',lw=1); axes[0].set(title='ROC', xlabel='FPR', ylabel='TPR')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    best_name = max(ml_res, key=lambda k: ml_res[k]['auc'])
    cm = confusion_matrix(y_te, ml_res[best_name]['yp'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Spoof','Live'], yticklabels=['Spoof','Live'])
    axes[1].set(title=f'Confusion Matrix ({best_name})', xlabel='Predicted', ylabel='Actual')
    plt.tight_layout(); plt.savefig('ml_evaluation.png', dpi=110, bbox_inches='tight')
    print('  Saved: ml_evaluation.png')

    # ── CNN training — two-stage fine-tuning ──────────────────
    STAGE1_EPOCHS = 5    # frozen backbone: train classifier head only
    STAGE2_EPOCHS = 10   # unfrozen: fine-tune full network at lower LR
    TOTAL_EPOCHS  = STAGE1_EPOCHS + STAGE2_EPOCHS
    BATCH         = 32

    print(f'\n[3/3] Training CNN ({TOTAL_EPOCHS} ep, {IMG_SIZE}x{IMG_SIZE}, {DEVICE})')
    print(f'      Stage 1: {STAGE1_EPOCHS} ep frozen  →  Stage 2: {STAGE2_EPOCHS} ep unfrozen')

    train_ds = SpoofDataset(train_s, TRAIN_TRANSFORM)
    val_ds   = SpoofDataset(val_s,   VAL_TRANSFORM)
    test_ds  = SpoofDataset(test_s,  VAL_TRANSFORM)
    sampler  = WeightedRandomSampler(train_ds.class_weights(), len(train_s), True)
    tr_ld = DataLoader(train_ds, BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    va_ld = DataLoader(val_ds,   BATCH, shuffle=False,   num_workers=2, pin_memory=True)
    te_ld = DataLoader(test_ds,  BATCH, shuffle=False,   num_workers=2, pin_memory=True)

    cnn  = LivenessCNN(pretrained=True, freeze_stages=True).to(DEVICE)
    crit = FocalBCELoss(gamma=2.0, alpha=0.75)

    def make_opt(lr, wd=1e-4):
        return optim.AdamW(filter(lambda p: p.requires_grad, cnn.parameters()), lr=lr, weight_decay=wd)

    hist = {'tl':[], 'vl':[], 'ta':[], 'va':[], 'vauc':[]}
    best_auc_cnn = 0

    def train_ep(loader, optimizer):
        cnn.train(); tl = tc = tt = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = cnn(imgs)
            loss   = crit(logits, lbls); loss.backward(); optimizer.step()
            tl += loss.item()
            tc += ((torch.sigmoid(logits.detach()) > 0.5).float() == lbls).sum().item()
            tt += len(lbls)
        return tl / len(loader), tc / tt

    @torch.no_grad()
    def eval_ep(loader):
        cnn.eval(); tl = tc = tt = 0; probs = []; lblall = []
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = cnn(imgs); loss = crit(logits, lbls)
            p = torch.sigmoid(logits)
            tl += loss.item(); tc += ((p > 0.5).float() == lbls).sum().item(); tt += len(lbls)
            probs.extend(p.cpu().numpy()); lblall.extend(lbls.cpu().numpy())
        auc = roc_auc_score(lblall, probs) if len(set(lblall)) > 1 else 0.0
        return tl / len(loader), tc / tt, auc

    # ── Stage 1: frozen backbone ────────────────────────────
    print(f'\n  ── Stage 1 (frozen backbone, lr=1e-3) ──')
    opt1 = make_opt(lr=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS)
    for ep in range(1, STAGE1_EPOCHS + 1):
        t0           = time.time()
        tl, ta       = train_ep(tr_ld, opt1)
        vl, va, vauc = eval_ep(va_ld)
        sch1.step()
        hist['tl'].append(tl); hist['vl'].append(vl)
        hist['ta'].append(ta); hist['va'].append(va); hist['vauc'].append(vauc)
        star = ''
        if vauc > best_auc_cnn:
            best_auc_cnn = vauc; torch.save(cnn.state_dict(), MODEL_PATH); star = ' ★'
        print(f'  S1 ep {ep}/{STAGE1_EPOCHS} | loss {tl:.4f}/{vl:.4f} | '
              f'acc {ta:.3f}/{va:.3f} | AUC {vauc:.3f}{star}  ({time.time()-t0:.0f}s)')

    # ── Stage 2: unfreeze all, fine-tune at low LR ──────────
    print(f'\n  ── Stage 2 (full fine-tune, lr=2e-4) ──')
    cnn.unfreeze_all()
    opt2 = make_opt(lr=2e-4, wd=5e-5)
    sch2 = optim.lr_scheduler.OneCycleLR(
        opt2, max_lr=2e-4, epochs=STAGE2_EPOCHS, steps_per_epoch=len(tr_ld),
        pct_start=0.2, anneal_strategy='cos')
    for ep in range(1, STAGE2_EPOCHS + 1):
        t0           = time.time()
        tl, ta       = train_ep(tr_ld, opt2)
        vl, va, vauc = eval_ep(va_ld)
        sch2.step()
        hist['tl'].append(tl); hist['vl'].append(vl)
        hist['ta'].append(ta); hist['va'].append(va); hist['vauc'].append(vauc)
        star = ''
        if vauc > best_auc_cnn:
            best_auc_cnn = vauc; torch.save(cnn.state_dict(), MODEL_PATH); star = ' ★'
        print(f'  S2 ep {ep}/{STAGE2_EPOCHS} | loss {tl:.4f}/{vl:.4f} | '
              f'acc {ta:.3f}/{va:.3f} | AUC {vauc:.3f}{star}  ({time.time()-t0:.0f}s)')

    cnn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, te_acc, te_auc = eval_ep(te_ld)
    print(f'\n  Test: Acc={te_acc:.3f} | AUC={te_auc:.3f}')
    print(f'  Saved: {MODEL_PATH}')
    print(f'\nTraining complete!')
    return cnn, best_ml, scaler


def load_model():
    m = LivenessCNN(pretrained=False, freeze_stages=False).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    print(f'Model loaded: {MODEL_PATH}  (device={DEVICE})')
    return m


# ═══════════════════════════════════════════════════════════
# 4. ACTIVE LIVENESS CHALLENGE SYSTEM
# ═══════════════════════════════════════════════════════════

@dataclass
class ChallengeResult:
    challenge: str; passed: bool; confidence: float
    duration_sec: float; details: dict = field(default_factory=dict)


def _max_consecutive(buf, threshold, above=True):
    """Hitung frame berturut-turut terpanjang yang melewati threshold."""
    best = cur = 0
    for v in buf:
        ok = (v > threshold) if above else (v < threshold)
        if ok: cur += 1; best = max(best, cur)
        else:  cur = 0
    return best


class ActiveLivenessSystem:
    ALL_CHALLENGES = ['blink', 'turn_left', 'turn_right', 'nod_down', 'nod_up']

    THR = dict(
        EAR_CLOSED          = 0.21,
        DELTA_YAW           = 20.0,   # ≥20° lateral rotation
        DELTA_PITCH_DOWN    = 12.0,   # ≥12° pitch change for nod down (abs, either direction)
        DELTA_PITCH_UP      =  5.0,   # ≥5° upward tilt (solvePnP channel)
        DELTA_VR_UP         =  0.040, # vertical-ratio drop when looking up
        SUSTAINED_FRAMES    =  8,     # consecutive frames needed for most gestures
        SUSTAINED_NOD_DOWN  =  4,     # separate easy requirement for nod_down
        SUSTAINED_NOD_UP    =  4,     # easier requirement for nod_up (smaller movement)
        BLINK_FRAMES        =  2,
        EAR_OPEN            =  0.23,  # eye considered "open" — lower than 0.26 for small eyes
        CALIBRATION_SEC     =  3.0,
        SESSION_SEC         = 90.0,
        MIN_CONF            =  0.55,
        CNN_SPOOF_THR       =  0.50,   # per-challenge gate (raised from 0.40)
        CNN_PRECHECK_THR    =  0.50,   # post-calibration passive spoof gate
        MOTION_LIVENESS_THR =  0.18,   # micro-motion gate: below = likely still image
        # Distance gating — face height (forehead→chin) / frame height
        FACE_MIN_SIZE       =  0.13,  # < this = too far → block
        FACE_MAX_SIZE       =  0.80,  # > this = too close → block
        FACE_CENTER_MARGIN  =  0.30,  # max normalized offset from frame center
        FACE_MATCH_TOL      =  0.52,  # face_recognition distance; > this = different person
    )

    LABELS = dict(blink='Kedip Mata', turn_left='Hadap Kiri',
                  turn_right='Hadap Kanan', nod_down='Angguk Bawah', nod_up='Angkat Kepala')
    HINT   = dict(blink='kedip pelan & tahan sebentar',
                  turn_left='<-- putar kepala kiri jauh',
                  turn_right='putar kepala kanan jauh -->',
                  nod_down='v angguk kepala ke bawah',
                  nod_up='^ angkat muka ke atas')

    def __init__(self, n=3):
        self.n = n; self.token = None; self.t_start = None
        self.seq: List[str] = []; self.results: List[ChallengeResult] = []
        self.cnn_scores: List[float] = []
        self.yaw_baseline:   Optional[float] = None
        self.pitch_baseline: Optional[float] = None
        self.vr_baseline:    Optional[float] = None

    def start_session(self):
        self.token   = str(uuid.uuid4()); self.t_start = time.time()
        self.results = []; self.cnn_scores = []
        self.yaw_baseline = None; self.pitch_baseline = None; self.vr_baseline = None
        seed    = int(hashlib.sha256(self.token.encode()).hexdigest(), 16) % (2**32)
        # Blink is always first — a still image/photo cannot blink
        others  = [c for c in self.ALL_CHALLENGES if c != 'blink']
        rest    = random.Random(seed).sample(others, k=self.n - 1)
        self.seq = ['blink'] + rest
        print(f'\nSesi baru | {self.token[:8]}... | {[self.LABELS[c] for c in self.seq]}')

    def set_baseline(self, yaw_samples, pitch_samples, vr_samples=None):
        self.yaw_baseline   = float(np.median(yaw_samples))
        self.pitch_baseline = float(np.median(pitch_samples))
        self.vr_baseline    = float(np.median(vr_samples)) if vr_samples else 0.50
        print(f'Baseline: yaw={self.yaw_baseline:.1f}°  pitch={self.pitch_baseline:.1f}°  vr={self.vr_baseline:.3f}')

    def elapsed(self): return time.time() - self.t_start if self.t_start else 0
    def expired(self): return self.elapsed() > self.THR['SESSION_SEC']

    def check(self, challenge, ear_buf, yaw_buf, pitch_buf, vr_buf=None):
        T = self.THR; passed = False; conf = 0.0; details = {}
        S = T['SUSTAINED_FRAMES']

        yaw_base   = self.yaw_baseline   if self.yaw_baseline   is not None else float(np.median(yaw_buf)   if yaw_buf   else 0)
        pitch_base = self.pitch_baseline if self.pitch_baseline is not None else float(np.median(pitch_buf) if pitch_buf else 0)
        vr_base    = self.vr_baseline    if self.vr_baseline    is not None else 0.50

        if challenge == 'blink':
            ok, n = detect_blink(ear_buf,
                                 threshold=T['EAR_CLOSED'],
                                 open_threshold=T['EAR_OPEN'],
                                 consec_frames=int(T['BLINK_FRAMES']))
            std   = float(np.std(ear_buf)) if ear_buf else 0
            conf  = min(1.0, n/2 + std*5); passed = ok and n >= 1
            details = {'blinks': n, 'ear_std': round(std, 3)}

        elif challenge == 'turn_left':
            # flipped webcam: physical left → yaw increases in mirrored frame
            deltas    = [y - yaw_base for y in yaw_buf]
            frames_ok = _max_consecutive(deltas, T['DELTA_YAW'], above=True)
            max_delta = max(deltas) if deltas else 0
            passed    = frames_ok >= S
            conf      = min(1.0, frames_ok/S) * min(1.0, max_delta/T['DELTA_YAW'])
            details   = {'baseline': round(yaw_base,1), 'delta_left': round(max_delta,1),
                         'sustained': frames_ok, 'need': S}

        elif challenge == 'turn_right':
            # flipped webcam: physical right → yaw decreases in mirrored frame
            deltas    = [yaw_base - y for y in yaw_buf]
            frames_ok = _max_consecutive(deltas, T['DELTA_YAW'], above=True)
            max_delta = max(deltas) if deltas else 0
            passed    = frames_ok >= S
            conf      = min(1.0, frames_ok/S) * min(1.0, max_delta/T['DELTA_YAW'])
            details   = {'baseline': round(yaw_base,1), 'delta_right': round(max_delta,1),
                         'sustained': frames_ok, 'need': S}

        elif challenge == 'nod_down':
            S_dn      = T['SUSTAINED_NOD_DOWN']
            thr_p     = T['DELTA_PITCH_DOWN']
            # Use abs delta — works regardless of solvePnP pitch sign convention
            deltas    = [abs(p - pitch_base) for p in pitch_buf]
            frames_ok = _max_consecutive(deltas, thr_p, above=True)
            max_delta = max(deltas) if deltas else 0
            passed    = frames_ok >= S_dn
            conf      = min(1.0, frames_ok/S_dn) * min(1.0, max_delta/thr_p)
            details   = {'baseline': round(pitch_base,1), 'delta_down': round(max_delta,1),
                         'sustained': frames_ok, 'need': S_dn}

        elif challenge == 'nod_up':
            S_up   = T['SUSTAINED_NOD_UP']
            thr_p  = T['DELTA_PITCH_UP']
            thr_vr = T['DELTA_VR_UP']

            # Smooth pitch to reduce solvePnP noise before evaluating
            # pitch increases when looking up in this OpenCV setup
            p_smooth  = _smooth_series(pitch_buf, n=3)
            deltas_p  = [p - pitch_base for p in p_smooth]
            frames_p  = _max_consecutive(deltas_p, thr_p, above=True)
            max_p     = max(deltas_p) if deltas_p else 0.0

            # Secondary metric: vertical landmark ratio (more stable for small tilts)
            # vr_val decreases when looking up (chin rises toward nose in image)
            vr_list   = list(vr_buf) if vr_buf else []
            deltas_vr = [vr_base - vr for vr in vr_list]
            frames_vr = _max_consecutive(deltas_vr, thr_vr, above=True) if deltas_vr else 0
            max_vr    = max(deltas_vr) if deltas_vr else 0.0

            # Pass if EITHER channel is confident, with partial credit from the other
            passed    = (frames_p >= S_up) or (frames_vr >= S_up and frames_p >= S_up // 2)
            score_p   = min(1.0, frames_p / S_up) * min(1.0, max_p  / (thr_p  + 1e-9))
            score_vr  = (min(1.0, frames_vr / S_up) * min(1.0, max_vr / (thr_vr + 1e-9))
                         if deltas_vr else 0.0)
            conf      = max(score_p, 0.6 * score_p + 0.4 * score_vr)
            details   = {'pitch_delta': round(max_p, 1), 'vr_delta': round(max_vr, 3),
                         'frames_p': frames_p, 'frames_vr': frames_vr, 'need': S_up}

        if self.cnn_scores:
            avg = float(np.mean(self.cnn_scores[-30:]))
            details['cnn'] = round(avg, 2)
            if avg < T['CNN_SPOOF_THR']:
                passed = False; conf *= avg/T['CNN_SPOOF_THR']; details['cnn_reject'] = True

        conf = max(0.0, min(1.0, conf))
        if passed and conf < T['MIN_CONF']: passed = False
        return ChallengeResult(challenge, passed, conf, self.elapsed(), details)

    def verdict(self, motion_score=None):
        n_ok = sum(1 for r in self.results if r.passed)
        return {'liveness': n_ok == len(self.seq), 'passed': n_ok,
                'total': len(self.seq),
                'avg_cnn': round(float(np.mean(self.cnn_scores)) if self.cnn_scores else 1.0, 3),
                'motion': round(motion_score, 3) if motion_score is not None else None,
                'reason': 'challenges'}


# ═══════════════════════════════════════════════════════════
# 5. HUD DRAWING
# ═══════════════════════════════════════════════════════════

GREEN=(50,230,50); RED=(50,50,230); ORANGE=(30,165,255)
WHITE=(230,230,230); GRAY=(140,140,140); DARK=(20,20,20); YELLOW=(30,215,255)


def put(img, txt, pos, scale=0.65, color=WHITE, thick=1, bold=False):
    f = cv2.FONT_HERSHEY_SIMPLEX
    if bold: cv2.putText(img, txt, pos, f, scale, DARK, thick+3)
    cv2.putText(img, txt, pos, f, scale, color, thick)


def _draw_face_oval(frame, face_size, dist_ok, too_far, too_close):
    """Draw eKYC-style face oval guide + proximity bar on the right edge."""
    h, w = frame.shape[:2]
    cx, cy   = w // 2, int(h * 0.50)
    oval_rx  = int(w * 0.21)
    oval_ry  = int(h * 0.37)
    col      = GREEN if dist_ok else RED
    cv2.ellipse(frame, (cx, cy), (oval_rx, oval_ry), 0, 0, 360, col, 3)

    if not dist_ok:
        if too_far:
            label = 'Move closer'
        elif too_close:
            label = 'Step back'
        else:
            label = 'Center your face'
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        put(frame, label, (cx - tw // 2, cy + oval_ry + 34), scale=0.65, color=RED, thick=2, bold=True)

    # Proximity bar — right edge
    T       = ActiveLivenessSystem.THR
    bx      = w - 28; by = h // 2 - 85; bh = 170
    cv2.rectangle(frame, (bx, by), (bx + 14, by + bh), (55, 55, 55), -1)
    # Green "ideal zone" band (35–65 % up the bar)
    iz1 = by + int(bh * 0.35); iz2 = by + int(bh * 0.65)
    cv2.rectangle(frame, (bx, iz1), (bx + 14, iz2), (30, 100, 30), -1)
    pct   = max(0.0, min(1.0, (face_size - T['FACE_MIN_SIZE']) /
                              (T['FACE_MAX_SIZE'] - T['FACE_MIN_SIZE'] + 1e-9)))
    dot_y = by + int(bh * (1.0 - pct))
    cv2.circle(frame, (bx + 7, dot_y), 8, col, -1)
    put(frame, 'FAR',  (bx - 2, by + bh + 14), scale=0.32, color=GRAY)
    put(frame, 'NEAR', (bx - 4, by - 5),        scale=0.32, color=GRAY)


def draw_hud(frame, sys, ch_idx, challenge, ear, yaw, pitch, cnn_s, passed_list,
             final_v=None, gesture_delta=0.0, calib_progress=None,
             face_size=0.0, too_far=False, too_close=False, dist_ok=True,
             motion_score=0.0, face_dist=0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,130), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Face oval is drawn in all phases
    _draw_face_oval(frame, face_size, dist_ok, too_far, too_close)

    put(frame, f'Sesi: {sys.token[:8]}', (12,22), scale=0.42, color=GRAY)

    # ── FASE KALIBRASI ──────────────────────────────────────
    if calib_progress is not None:
        face_ok = ear > 0 and dist_ok
        if not ear:
            sub = 'Position your face in the oval'
        elif too_far:
            sub = 'Move closer to the camera'
        elif too_close:
            sub = 'Step back from the camera'
        elif not dist_ok:
            sub = 'Center your face in the oval'
        else:
            secs_left = ActiveLivenessSystem.THR['CALIBRATION_SEC'] * (1 - calib_progress)
            sub = f'Hold still... {secs_left:.1f}s'
        put(frame, 'Look Straight at Camera', (12,65), scale=0.85, color=YELLOW, thick=2, bold=True)
        put(frame, sub, (12,95), scale=0.55, color=WHITE if face_ok else ORANGE)

        bw = w - 24
        cv2.rectangle(frame, (12,112), (w-12,126), (60,60,60), -1)
        fill = int(bw * calib_progress) if face_ok else 0
        cv2.rectangle(frame, (12,112), (12+fill,126), GREEN if face_ok else GRAY, -1)
        put(frame, f'Yaw {yaw:+.1f}°   Pitch {pitch:+.1f}°', (12, h-55), scale=0.48, color=GRAY)

    # ── FASE CHALLENGE ──────────────────────────────────────
    elif final_v is None:
        ses_left = max(0, sys.THR['SESSION_SEC'] - sys.elapsed())
        put(frame, f'{ses_left:.0f}s left', (w-110,22), scale=0.42,
            color=GREEN if ses_left>30 else ORANGE if ses_left>10 else RED)

        for i, _ in enumerate(sys.seq):
            col = GREEN if i < len(passed_list) else YELLOW if i == ch_idx else GRAY
            cv2.circle(frame, (12 + i*30, 42), 10, col, -1)
            cv2.circle(frame, (12 + i*30, 42), 10, DARK, 1)

        if not dist_ok:
            dist_msg = 'Too far — move closer' if too_far else 'Too close — step back'
            put(frame, dist_msg, (12, 75), scale=0.85, color=RED, thick=2, bold=True)
            put(frame, 'Challenge paused', (12, 100), scale=0.55, color=ORANGE)
        else:
            put(frame, sys.LABELS.get(challenge,''), (12,75), scale=0.95, color=YELLOW, thick=2, bold=True)
            put(frame, sys.HINT.get(challenge,''),   (12,100), scale=0.55, color=WHITE)

        bw = w - 24
        cv2.rectangle(frame, (12,112), (w-12,126), (60,60,60), -1)
        cv2.rectangle(frame, (12,112), (12+int(bw*ses_left/sys.THR['SESSION_SEC']),126),
                      GREEN if ses_left>30 else ORANGE if ses_left>10 else RED, -1)

        cnn_col  = GREEN if cnn_s > 0.55 else ORANGE if cnn_s > 0.5 else RED
        mot_col  = GREEN if motion_score > 0.25 else ORANGE if motion_score > 0.12 else RED
        face_col = GREEN if face_dist < 0.40 else ORANGE if face_dist < 0.52 else RED
        for i, (txt, col) in enumerate([
            (f'EAR  {ear:.3f}',          GREEN if ear > 0.21 else RED),
            (f'Yaw  {yaw:+.1f}\xb0',     WHITE),
            (f'Ptch {pitch:+.1f}\xb0',   WHITE),
            (f'CNN  {cnn_s:.2f}',         cnn_col),
            (f'Mot  {motion_score:.2f}',  mot_col),
            (f'Face {face_dist:.2f}',     face_col),
        ]):
            put(frame, txt, (w-150, h-132+i*22), scale=0.5, color=col)

        T = ActiveLivenessSystem.THR
        bar_y = h - 55; bar_x = 12; bar_w = 220; bar_h = 14
        if challenge in ('turn_left', 'turn_right'):
            thr_v = T['DELTA_YAW']
            lbl   = f'Delta {"kiri" if challenge=="turn_left" else "kanan"}: {gesture_delta:.1f}° / {thr_v:.0f}°'
            val   = gesture_delta
        elif challenge in ('nod_down', 'nod_up'):
            thr_v = T['DELTA_PITCH_DOWN'] if challenge == 'nod_down' else T['DELTA_PITCH_UP']
            lbl   = f'Delta {"bawah" if challenge=="nod_down" else "atas"}: {gesture_delta:.1f}° / {thr_v:.0f}°'
            val   = gesture_delta
        else:
            val = max(0, T['EAR_CLOSED'] - ear) * 20; thr_v = T['EAR_CLOSED'] * 20
            lbl = f'EAR {ear:.3f}  (target <{T["EAR_CLOSED"]})'
        pct = min(1.0, max(0, val) / (thr_v + 1e-6))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
        fill_col = GREEN if pct >= 1.0 else ORANGE if pct > 0.6 else RED
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*pct), bar_y+bar_h), fill_col, -1)
        put(frame, lbl, (bar_x, bar_y-4), scale=0.42, color=WHITE)

    # ── HASIL AKHIR ─────────────────────────────────────────
    else:
        msg = 'LIVENESS VERIFIED' if final_v['liveness'] else 'SPOOF DETECTED'
        col = GREEN if final_v['liveness'] else RED
        put(frame, msg, (12,80), scale=1.1, color=col, thick=3, bold=True)
        reason = final_v.get('reason', '')
        if reason == 'passive_spoof':
            put(frame, 'Still image / photo detected', (12, 112), scale=0.55, color=RED)
        elif reason == 'low_motion':
            put(frame, 'No face micro-motion detected', (12, 112), scale=0.55, color=RED)
        elif reason == 'face_mismatch':
            put(frame, 'Face does not match calibration', (12, 112), scale=0.55, color=RED)
        else:
            mot_s = final_v.get('motion')
            mot_str = f'  Motion={mot_s:.2f}' if mot_s is not None else ''
            put(frame, f"Passed {final_v['passed']}/{final_v['total']}  CNN={final_v['avg_cnn']}{mot_str}",
                (12,112), scale=0.55, color=WHITE)

    if ear == 0:
        put(frame, 'Wajah tidak terdeteksi', (12, h-20), scale=0.52, color=RED)
    if final_v is not None and not final_v['liveness']:
        put(frame, 'Tunjukkan wajah asli — sesi baru otomatis...', (12, h-20), scale=0.42, color=ORANGE)
    put(frame, '[Q] Keluar  [S] Sesi Baru', (12, h-8), scale=0.40, color=GRAY)
    return frame


# ═══════════════════════════════════════════════════════════
# 6. WEBCAM DEMO
# ═══════════════════════════════════════════════════════════

def run_webcam(cnn_model, n_challenges=3, camera_idx=0):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Kamera tidak ditemukan!'); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    CALIB_SEC = ActiveLivenessSystem.THR['CALIBRATION_SEC']

    def _reset():
        s = ActiveLivenessSystem(n=n_challenges)
        s.start_session()
        return (s, 0,
                deque(maxlen=40), deque(maxlen=40), deque(maxlen=40), deque(maxlen=40),
                [], None,
                True, [], None)   # calibrating, calib_yaw, calib_t0

    sys, ch_idx, ear_buf, yaw_buf, pit_buf, vr_buf, passed, final_v, \
        calibrating, calib_yaw, calib_t0 = _reset()
    calib_pitch    = []
    calib_vr       = []
    calib_ear      = []        # EAR values during calibration (for motion score)
    calib_nose     = []        # nose tip (x,y) normalized during calibration
    motion_score   = 0.0       # updated after calibration gate, displayed in HUD
    calib_encoding = None      # 128-dim face encoding locked at end of calibration
    face_dist      = 0.0       # latest face distance vs calibration (0=match, 1=no match)
    enc_frame_ctr  = 0         # counts frames; face encoding checked every 20 frames
    verdict_time   = None      # timestamp when final_v was set (for auto-reset)

    print('\nKamera aktif. Q=keluar  S=sesi baru\n')

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        ear = yaw = pitch = 0.0
        face_size = 0.0; too_far = True; too_close = False; dist_ok = False
        vr_val = 0.50; nose_xy = (0.5, 0.5); face_bbox_px = None

        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_detected = bool(res.multi_face_landmarks)
        if face_detected:
            lm    = res.multi_face_landmarks[0].landmark
            ear   = (compute_EAR(lm, LEFT_EYE_IDX, w, h) +
                     compute_EAR(lm, RIGHT_EYE_IDX, w, h)) / 2
            yaw, pitch, _ = get_head_pose(lm, w, h)
            vr_val        = compute_vertical_ratio(lm, h)
            nose_xy       = (lm[4].x, lm[4].y)   # nose tip normalized coords

            # Distance / proximity check
            T_thr      = ActiveLivenessSystem.THR
            face_size  = compute_face_size(lm, w, h)
            face_cx, face_cy = compute_face_center(lm)
            too_far    = face_size < T_thr['FACE_MIN_SIZE']
            too_close  = face_size > T_thr['FACE_MAX_SIZE']
            off_center = (abs(face_cx - 0.5) > T_thr['FACE_CENTER_MARGIN'] or
                          abs(face_cy - 0.5) > T_thr['FACE_CENTER_MARGIN'])
            dist_ok    = not too_far and not too_close and not off_center

            # Compute face bounding box from all landmarks for CNN crop
            xs = [lm[i].x * w for i in range(len(lm))]
            ys = [lm[i].y * h for i in range(len(lm))]
            pad = int(0.15 * (max(ys) - min(ys)))
            x1 = max(0, int(min(xs)) - pad); y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad); y2 = min(h, int(max(ys)) + pad)
            face_bbox_px = (x1, y1, x2, y2)

        cnn_s = score_frame(cnn_model, frame, face_bbox=face_bbox_px)
        sys.cnn_scores.append(cnn_s)

        # ── FASE KALIBRASI ─────────────────────────────────
        if calibrating:
            if face_detected and dist_ok:
                if calib_t0 is None:
                    calib_t0 = time.time()
                calib_yaw.append(yaw); calib_pitch.append(pitch); calib_vr.append(vr_val)
                calib_ear.append(ear); calib_nose.append(nose_xy)
                elapsed_c = time.time() - calib_t0
                calib_pct = min(1.0, elapsed_c / CALIB_SEC)

                # Early reject: if first ~25 frames already look like a spoof, don't wait 3s
                if len(calib_nose) == 25:
                    early_cnn = float(np.mean(sys.cnn_scores[-25:]))
                    if early_cnn < 0.35:
                        final_v = {'liveness': False, 'passed': 0,
                                   'total': len(sys.seq),
                                   'avg_cnn': round(early_cnn, 3),
                                   'motion': 0.0, 'reason': 'passive_spoof'}
                        calibrating = False
                        print(f'VERDICT: SPOOF DETECTED ✗ (early CNN={early_cnn:.3f})')
                if elapsed_c >= CALIB_SEC:
                    sys.set_baseline(calib_yaw, calib_pitch, calib_vr)
                    calibrating = False

                    # ── PASSIVE SPOOF PRE-CHECK ────────────────────
                    T_thr        = ActiveLivenessSystem.THR
                    n_c          = max(10, len(sys.cnn_scores))
                    cnn_calib    = float(np.mean(sys.cnn_scores[-n_c:]))
                    motion_score = compute_motion_score(calib_nose, calib_ear)

                    if cnn_calib < T_thr['CNN_PRECHECK_THR']:
                        final_v = {'liveness': False, 'passed': 0,
                                   'total': len(sys.seq),
                                   'avg_cnn': round(cnn_calib, 3),
                                   'motion': round(motion_score, 3),
                                   'reason': 'passive_spoof'}
                        print(f'VERDICT: SPOOF DETECTED ✗ (CNN pre-check={cnn_calib:.3f})')
                    elif motion_score < T_thr['MOTION_LIVENESS_THR']:
                        final_v = {'liveness': False, 'passed': 0,
                                   'total': len(sys.seq),
                                   'avg_cnn': round(cnn_calib, 3),
                                   'motion': round(motion_score, 3),
                                   'reason': 'low_motion'}
                        print(f'VERDICT: SPOOF DETECTED ✗ (motion pre-check={motion_score:.3f})')
                    else:
                        # Lock in the face encoding for this session
                        calib_encoding = extract_face_encoding(frame)
                        if calib_encoding is None:
                            print('Warning: could not extract face encoding at calibration end.')
                        print(f'Pre-check OK: CNN={cnn_calib:.3f}  Motion={motion_score:.3f}')
                        print('Starting challenges.\n')
            else:
                # Reset timer if face disappears or goes out of range
                calib_t0 = None
                calib_yaw.clear(); calib_pitch.clear(); calib_vr.clear()
                calib_ear.clear(); calib_nose.clear()
                calib_pct = 0.0

            frame = draw_hud(frame, sys, 0, '', ear, yaw, pitch, cnn_s, [],
                             calib_progress=calib_pct,
                             face_size=face_size, too_far=too_far,
                             too_close=too_close, dist_ok=dist_ok,
                             motion_score=motion_score)
            cv2.imshow('e-KYC Active Liveness  [Q=quit  S=new session]', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('s'):
                sys, ch_idx, ear_buf, yaw_buf, pit_buf, vr_buf, passed, final_v, \
                    calibrating, calib_yaw, calib_t0 = _reset()
                calib_pitch = []; calib_vr = []; calib_ear = []; calib_nose = []
                motion_score = 0.0; calib_encoding = None; face_dist = 0.0; enc_frame_ctr = 0
            continue

        # ── FASE CHALLENGE ─────────────────────────────────
        if face_detected and dist_ok and final_v is None and ch_idx < len(sys.seq):
            ear_buf.append(ear); yaw_buf.append(yaw)
            pit_buf.append(pitch); vr_buf.append(vr_val)
        elif face_detected and not dist_ok:
            ear_buf.clear(); yaw_buf.clear(); pit_buf.clear(); vr_buf.clear()

        # ── FACE IDENTITY CHECK (every 20 frames) ──────────
        if final_v is None and calib_encoding is not None:
            enc_frame_ctr += 1
            if enc_frame_ctr % 20 == 0 and face_detected:
                cur_enc = extract_face_encoding(frame)
                if cur_enc is not None:
                    face_dist = float(face_recognition.face_distance(
                        [calib_encoding], cur_enc)[0])
                    T_thr = ActiveLivenessSystem.THR
                    if face_dist > T_thr['FACE_MATCH_TOL']:
                        final_v = {'liveness': False, 'passed': len(passed),
                                   'total': len(sys.seq),
                                   'avg_cnn': round(float(np.mean(sys.cnn_scores)) if sys.cnn_scores else 0, 3),
                                   'motion': round(motion_score, 3),
                                   'reason': 'face_mismatch'}
                        print(f'VERDICT: SPOOF DETECTED ✗ (face distance={face_dist:.3f})')

        if final_v is None and sys.expired():
            final_v = {'liveness': False, 'passed': len(passed),
                       'total': len(sys.seq), 'avg_cnn': 0.0}
            print('VERDICT: LIVENESS FAILED ✗ (session expired)')

        if final_v is None and ch_idx < len(sys.seq) and len(ear_buf) >= 12:
            challenge = sys.seq[ch_idx]
            result    = sys.check(challenge, list(ear_buf), list(yaw_buf),
                                  list(pit_buf), vr_buf=list(vr_buf))
            if result.passed:
                sys.results.append(result); passed.append(True)
                print(f'[✓] {sys.LABELS[challenge]:14s} conf={result.confidence:.2f}  {result.details}')
                ch_idx += 1
                ear_buf.clear(); yaw_buf.clear(); pit_buf.clear(); vr_buf.clear()
                if ch_idx >= len(sys.seq):
                    final_v = sys.verdict(motion_score=motion_score)
                    print(f'\nVERDICT: LIVENESS VERIFIED ✓')
                    print(f"Passed {final_v['passed']}/{final_v['total']} | CNN={final_v['avg_cnn']} | Motion={final_v['motion']}\n")

        ch_now = sys.seq[ch_idx] if ch_idx < len(sys.seq) else ''
        if ch_now in ('turn_left', 'turn_right'):
            _base = sys.yaw_baseline if sys.yaw_baseline is not None else yaw
            g_delta = (yaw - _base) if ch_now == 'turn_left' else (_base - yaw)
        elif ch_now == 'nod_down':
            _base = sys.pitch_baseline if sys.pitch_baseline is not None else pitch
            g_delta = abs(pitch - _base)
        elif ch_now == 'nod_up':
            # pitch increases when looking up; show the stronger of the two signals
            _base_p  = sys.pitch_baseline if sys.pitch_baseline is not None else pitch
            _base_vr = sys.vr_baseline    if sys.vr_baseline    is not None else vr_val
            g_delta  = max(pitch - _base_p,
                           (_base_vr - vr_val) / (ActiveLivenessSystem.THR['DELTA_VR_UP'] + 1e-9)
                           * ActiveLivenessSystem.THR['DELTA_PITCH_UP'])
        else:
            g_delta = 0.0

        # Track when verdict was set; auto-reset on SPOOF after 4 s if real face shown
        if final_v is not None and verdict_time is None:
            verdict_time = time.time()
        if (final_v is not None and not final_v['liveness']
                and verdict_time is not None
                and time.time() - verdict_time >= 4.0
                and face_detected and dist_ok):
            sys, ch_idx, ear_buf, yaw_buf, pit_buf, vr_buf, passed, final_v, \
                calibrating, calib_yaw, calib_t0 = _reset()
            calib_pitch = []; calib_vr = []; calib_ear = []; calib_nose = []
            motion_score = 0.0; calib_encoding = None; face_dist = 0.0
            enc_frame_ctr = 0; verdict_time = None

        frame = draw_hud(frame, sys, ch_idx, ch_now, ear, yaw, pitch, cnn_s, passed,
                         final_v, gesture_delta=g_delta,
                         face_size=face_size, too_far=too_far,
                         too_close=too_close, dist_ok=dist_ok,
                         motion_score=motion_score, face_dist=face_dist)
        cv2.imshow('e-KYC Active Liveness  [Q=quit  S=new session]', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('s'):
            sys, ch_idx, ear_buf, yaw_buf, pit_buf, vr_buf, passed, final_v, \
                calibrating, calib_yaw, calib_t0 = _reset()
            calib_pitch = []; calib_vr = []; calib_ear = []; calib_nose = []
            motion_score = 0.0; calib_encoding = None; face_dist = 0.0
            enc_frame_ctr = 0; verdict_time = None

    cap.release(); mesh.close(); cv2.destroyAllWindows()
    print('Demo selesai.')


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', action='store_true', help='Force retrain even if model exists')
    args = ap.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Model : {MODEL_PATH}')

    if args.retrain and MODEL_PATH.exists():
        MODEL_PATH.unlink()
        print('Existing model deleted — retraining...')

    if MODEL_PATH.exists():
        print(f'\nModel ditemukan — skip training, langsung buka kamera.')
        cnn_model = load_model()
    else:
        cnn_model, _, _ = run_training()
        if cnn_model is None:
            print('Training gagal. Periksa dataset.'); exit(1)

    run_webcam(cnn_model, n_challenges=5)
