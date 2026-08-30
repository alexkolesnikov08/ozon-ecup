#!/usr/bin/env python3
import time, pathlib, math
from datetime import date, timedelta
import numpy as np, polars as pl
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log(f"device {device}")

DATA=pl.read_parquet("data/train.parquet").with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64), pl.col("searches").cast(pl.Float64), pl.col("to_ord").cast(pl.Float64), pl.col("to_cart").cast(pl.Float64))
SAMPLE_N=1000
all_uids=DATA["user_id"].unique().sort().to_list()[:SAMPLE_N]
ANCHORS_TRAIN=[date(2025,11,26),date(2025,12,3),date(2025,12,10),date(2025,12,17),date(2025,12,24),date(2025,12,31),date(2026,1,7)]
ANCHOR_TEST=date(2026,1,14)
SEQ_LEN=90
N_CH=4

def build_seq(anchor, uids):
    # returns dict uid -> seq (90x4) and target
    # Build for all uids at once via cross join similar to lag
    dates=pl.date_range(anchor-timedelta(days=SEQ_LEN-1), anchor, "1d", eager=True)
    grid=pl.DataFrame({"user_id":uids}).join(pl.DataFrame({"event_date":dates}), how="cross")
    win=DATA.filter(pl.col("event_date").is_between(anchor-timedelta(days=SEQ_LEN-1), anchor)).filter(pl.col("user_id").is_in(uids)).select(["user_id","event_date","gmv","searches","to_ord","to_cart"])
    df=grid.join(win, on=["user_id","event_date"], how="left").with_columns([pl.col("gmv").fill_null(0.0),pl.col("searches").fill_null(0.0),pl.col("to_ord").fill_null(0.0),pl.col("to_cart").fill_null(0.0)]).sort(["user_id","event_date"])
    # group to seq
    grouped=df.group_by("user_id").agg([pl.col("gmv").alias("gmv_seq"),pl.col("searches").alias("s_seq"),pl.col("to_ord").alias("o_seq"),pl.col("to_cart").alias("c_seq")])
    # target
    tgt=DATA.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=30))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    idx=pl.DataFrame({"user_id":uids})
    tgt_df=idx.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0)).sort("user_id")
    # build mat
    sorted_uids=sorted(uids)
    user_to_seq={uid: (g,s,o,c) for uid,g,s,o,c in zip(grouped["user_id"].to_list(), grouped["gmv_seq"].to_list(), grouped["s_seq"].to_list(), grouped["o_seq"].to_list(), grouped["c_seq"].to_list())}
    seqs=[]
    for uid in sorted_uids:
        g,s,o,c=user_to_seq[uid]
        # g,s,o,c are lists length 90 oldest->newest
        arr=np.stack([np.array(g, dtype=np.float32), np.array(s, dtype=np.float32), np.array(o, dtype=np.float32), np.array(c, dtype=np.float32)], axis=1) # 90x4
        # log1p for stability
        arr=np.log1p(arr)
        seqs.append(arr)
    seqs=np.stack(seqs) # N x 90 x4
    y=np.log1p(tgt_df["target"].to_numpy().astype(float))
    return seqs, y, sorted_uids

log("Building train seqs ...")
train_seqs=[]; train_ys=[]
for a in ANCHORS_TRAIN:
    s,y,_=build_seq(a, all_uids)
    train_seqs.append(s); train_ys.append(y)
    log(f" anchor {a} {s.shape}")
X_train=np.concatenate(train_seqs, axis=0) # 7000 x90x4
y_train=np.concatenate(train_ys, axis=0)
log(f"X_train {X_train.shape} y {y_train.shape}")
X_test,y_test,_=build_seq(ANCHOR_TEST, all_uids)
log(f"X_test {X_test.shape}")

# Dataset
class SeqDataset(Dataset):
    def __init__(self, X, y): self.X=torch.tensor(X, dtype=torch.float32); self.y=torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds=SeqDataset(X_train, y_train)
test_ds=SeqDataset(X_test, y_test)
train_loader=DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader=DataLoader(test_ds, batch_size=1024)

# Model ~300k params
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=90):
        super().__init__()
        pe=torch.zeros(max_len, d_model)
        pos=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)
    def forward(self, x): # x B x L x D
        return x + self.pe[:x.size(1)]

class TransRegressor(nn.Module):
    def __init__(self, n_ch=4, d_model=64, nhead=4, nlayers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.input_proj=nn.Linear(n_ch, d_model)
        self.pos=PosEnc(d_model)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head=nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64,1))
    def forward(self, x): # B,L,4
        h=self.input_proj(x) # B,L,D
        h=self.pos(h)
        h=self.encoder(h) # B,L,D
        h=h.mean(dim=1) # GAP
        return self.head(h).squeeze(-1)

model=TransRegressor().to(device)
log(f"Params {sum(p.numel() for p in model.parameters())}")
opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
crit=nn.MSELoss()

best=1e9
for epoch in range(30):
    model.train()
    tloss=0
    for xb,yb in train_loader:
        xb,yb=xb.to(device), yb.to(device)
        opt.zero_grad()
        pred=model(xb)
        loss=crit(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tloss+=loss.item()*len(yb)
    sched.step()
    # val
    model.eval()
    with torch.no_grad():
        preds=[]; trues=[]
        for xb,yb in test_loader:
            xb=xb.to(device)
            p=model(xb).cpu().numpy()
            preds.append(p); trues.append(yb.numpy())
        preds=np.concatenate(preds); trues=np.concatenate(trues)
        rmsle=np.sqrt(mean_squared_error(trues, preds))
        tloss/=len(train_ds)
        log(f"Epoch {epoch+1:02d} train {tloss:.4f} val RMSLE {rmsle:.5f} lr {opt.param_groups[0]['lr']:.2e}")
        if rmsle<best:
            best=rmsle
            torch.save(model.state_dict(), "reports/transformer_best.pt")
            log(f"  new best {best:.5f}")

log(f"BEST {best:.5f} vs agg 1.67")
# also compare to lag baseline 1.86
