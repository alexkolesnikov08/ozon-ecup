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
all_uids=DATA["user_id"].unique().sort().to_list()
log(f"Users {len(all_uids)}")
ANCHORS_TRAIN=[date(2025,11,26),date(2025,12,3),date(2025,12,10),date(2025,12,17),date(2025,12,24),date(2025,12,31),date(2026,1,7)]
ANCHOR_TEST=date(2026,1,14)
ANCHOR_SUBMIT=date(2026,2,13)
SEQ_LEN=90
N_CH=4
BATCH_USERS=10000

def build_seq_batched(anchors, uids, is_train=True):
    # returns X (N*len(anchors) x90x4), y (N*len)
    # Process uids in batches to avoid huge cross join
    all_X=[]; all_y=[]
    for bi in range(0, len(uids), BATCH_USERS):
        batch_uids=uids[bi:bi+BATCH_USERS]
        log(f" batch {bi//BATCH_USERS+1}/{(len(uids)+BATCH_USERS-1)//BATCH_USERS} {len(batch_uids)} users")
        for anchor in anchors:
            dates=pl.date_range(anchor-timedelta(days=SEQ_LEN-1), anchor, "1d", eager=True)
            grid=pl.DataFrame({"user_id":batch_uids}).join(pl.DataFrame({"event_date":dates}), how="cross")
            win=DATA.filter(pl.col("event_date").is_between(anchor-timedelta(days=SEQ_LEN-1), anchor)).filter(pl.col("user_id").is_in(batch_uids)).select(["user_id","event_date","gmv","searches","to_ord","to_cart"])
            df=grid.join(win, on=["user_id","event_date"], how="left").with_columns([pl.col("gmv").fill_null(0.0),pl.col("searches").fill_null(0.0),pl.col("to_ord").fill_null(0.0),pl.col("to_cart").fill_null(0.0)]).sort(["user_id","event_date"])
            grouped=df.group_by("user_id").agg([pl.col("gmv").alias("gmv_seq"),pl.col("searches").alias("s_seq"),pl.col("to_ord").alias("o_seq"),pl.col("to_cart").alias("c_seq")]).sort("user_id")
            # target
            tgt=DATA.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=30))).filter(pl.col("user_id").is_in(batch_uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
            idx=pl.DataFrame({"user_id":batch_uids}).sort("user_id")
            tgt_df=idx.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0)).sort("user_id")
            # build array
            # ensure order matches
            # grouped is sorted, idx sorted
            seqs=[]
            for row in grouped.iter_rows(named=True):
                g=np.array(row["gmv_seq"], dtype=np.float32)
                s=np.array(row["s_seq"], dtype=np.float32)
                o=np.array(row["o_seq"], dtype=np.float32)
                c=np.array(row["c_seq"], dtype=np.float32)
                arr=np.stack([g,s,o,c], axis=1) # 90x4
                arr=np.log1p(arr)
                seqs.append(arr)
            seqs=np.stack(seqs) # B x90x4
            y=np.log1p(tgt_df["target"].to_numpy().astype(float))
            all_X.append(seqs); all_y.append(y)
    X=np.concatenate(all_X, axis=0)
    y=np.concatenate(all_y, axis=0)
    return X, y

log("Building TRAIN 1.75M seqs ...")
X_train, y_train = build_seq_batched(ANCHORS_TRAIN, all_uids, is_train=True)
log(f"X_train {X_train.shape} {X_train.nbytes/1e9:.2f}GB")
log("Building TEST 250k ...")
X_test, y_test = build_seq_batched([ANCHOR_TEST], all_uids, is_train=False)
log(f"X_test {X_test.shape}")
log("Building SUBMIT 250k ...")
X_submit, _ = build_seq_batched([ANCHOR_SUBMIT], all_uids, is_train=False)
# For submit we need y dummy, but we built y for submit anchor (which is 0), ignore y
# Actually build_seq_batched for submit returns y for that anchor's target (which is future beyond data, 0), not needed
# X_submit is first 250k of X_submit? Our function returns concatenated for one anchor, so X_submit shape 250k x90x4
X_submit = X_submit[:250000]  # ensure
log(f"X_submit {X_submit.shape}")

class SeqDataset(Dataset):
    def __init__(self, X, y): self.X=torch.tensor(X, dtype=torch.float32); self.y=torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds=SeqDataset(X_train, y_train)
test_ds=SeqDataset(X_test, y_test)
train_loader=DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
test_loader=DataLoader(test_ds, batch_size=4096, num_workers=0)

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=90):
        super().__init__()
        pe=torch.zeros(max_len, d_model)
        pos=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1)]

class TransRegressor(nn.Module):
    def __init__(self, n_ch=4, d_model=96, nhead=6, nlayers=3, d_ff=384, dropout=0.1):
        super().__init__()
        self.input_proj=nn.Linear(n_ch, d_model)
        self.pos=PosEnc(d_model)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head=nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64,1))
    def forward(self, x):
        h=self.input_proj(x)
        h=self.pos(h)
        h=self.encoder(h)
        h=h.mean(dim=1)
        return self.head(h).squeeze(-1)

model=TransRegressor().to(device)
log(f"Params {sum(p.numel() for p in model.parameters())}")
opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
crit=nn.MSELoss()
best=1e9
for epoch in range(15):
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
        log(f"Epoch {epoch+1:02d} train {tloss:.4f} val {rmsle:.5f} lr {opt.param_groups[0]['lr']:.2e}")
        if rmsle<best:
            best=rmsle
            torch.save(model.state_dict(), "reports/transformer_full_best.pt")
            log(f" new best {best:.5f}")

log(f"BEST {best:.5f}")
# Generate submit
model.load_state_dict(torch.load("reports/transformer_full_best.pt", map_location=device))
model.eval()
submit_ds=SeqDataset(X_submit, np.zeros(len(X_submit)))
submit_loader=DataLoader(submit_ds, batch_size=4096)
preds=[]
with torch.no_grad():
    for xb,_ in submit_loader:
        xb=xb.to(device)
        p=model(xb).cpu().numpy()
        preds.append(p)
preds=np.concatenate(preds)
preds_y=np.maximum(np.expm1(preds),0.0)
# Save to Desktop
import pathlib
out=pathlib.Path.home()/"Desktop"/"submission_transformer_full.csv"
pl.DataFrame({"user_id":sorted(all_uids), "predict":preds_y}).sort("user_id").write_csv(str(out))
log(f"Saved {out} mean {preds_y.mean():.2f}")
# also save report
import json
pathlib.Path("reports/transformer_full.json").write_text(json.dumps({"best_rmsle":float(best), "params":sum(p.numel() for p in model.parameters())}, indent=2))
