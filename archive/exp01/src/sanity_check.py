"""Sanity checks on built feature parquets + manual feature verification."""

from datetime import date, timedelta

import polars as pl

FEAT_COLS = None


def read_fold(name: str) -> pl.DataFrame:
    return pl.read_parquet(f"data/v2/features/{name}/batch_*.parquet")


def main() -> None:
    # 1) shape / nulls / target stats for every fold
    for name in ["fold_00", "fold_01", "fold_02", "fold_03", "fold_end"]:
        df = read_fold(name)
        n_null = df.null_count().row(0, named=True)
        total_nulls = sum(n_null.values())
        print(
            f"{name}: shape={df.shape}, nulls={total_nulls}, "
            f"anchors={df['anchor_date'].unique().to_list()}"
        )
        if name != "fold_end":
            t = df["target"]
            print(
                f"   target: zero_share={(t == 0).mean():.4f}, "
                f"mean={t.mean():.2f}, median={t.median():.2f}, max={t.max():.2f}"
            )
        else:
            assert df["target"].null_count() == df.height
            print("   target: all NULL as expected")

    f3 = read_fold("fold_03")
    global FEAT_COLS
    FEAT_COLS = [c for c in f3.columns if c not in ("anchor_date", "user_id", "target")]
    print(f"\n{len(FEAT_COLS)} feature cols:")
    print(FEAT_COLS)

    # 2) manual verification for one user on fold_03 (anchor 2026-01-14)
    a = date(2026, 1, 14)
    uid = f3["user_id"][0]
    row = f3.row(0, named=True)

    raw = pl.read_parquet("data/train.parquet").filter(pl.col("user_id") == uid)
    hist = raw.filter(pl.col("event_date").is_between(a - timedelta(days=89), a))

    def agg(col, fn):
        v = hist[col]
        return {"sum": v.sum(), "max": v.max(), "mean": v.mean()}[fn]

    checks = []
    for col in ("gmv", "searches", "to_ord", "to_cart"):
        for fn in ("sum", "max", "mean"):
            got, exp = row[f"{col}_{fn}_90d"], float(agg(col, fn))
            ok = abs(got - exp) < 1e-6 * max(1.0, abs(exp))
            checks.append((f"{col}_{fn}_90d", got, exp, ok))

    w30 = raw.filter(pl.col("event_date").is_between(a - timedelta(days=29), a))
    exp_active = w30.filter((pl.col("gmv") > 0) | (pl.col("searches") > 0)).height
    checks.append(("active_days_30d", row["active_days_30d"], float(exp_active),
                   abs(row["active_days_30d"] - exp_active) < 1e-9))

    h_all = raw.filter(pl.col("event_date") <= a)
    lo = h_all.filter(pl.col("to_ord") > 0)["event_date"].max()
    exp_rec = 999.0 if lo is None else float((a - lo).days)
    checks.append(("recency_to_ord_days", row["recency_to_ord_days"], exp_rec,
                   row["recency_to_ord_days"] == exp_rec))
    ls = h_all.filter(pl.col("searches") > 0)["event_date"].max()
    exp_rec_s = 999.0 if ls is None else float((a - ls).days)
    checks.append(("recency_searches_days", row["recency_searches_days"], exp_rec_s,
                   row["recency_searches_days"] == exp_rec_s))
    fe = h_all["event_date"].min()
    exp_ten = float((a - fe).days)
    checks.append(("tenure_days", row["tenure_days"], exp_ten,
                   row["tenure_days"] == exp_ten))

    tgt = raw.filter(pl.col("event_date").is_between(a + timedelta(days=1), a + timedelta(days=30)))
    exp_tgt = float(tgt["gmv"].sum())
    checks.append(("target", row["target"], exp_tgt, abs(row["target"] - exp_tgt) < 1e-6))

    bad = [c for c in checks if not c[3]]
    for name, got, exp, ok in checks:
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {name}: built={got} expected={exp}")
    assert not bad, f"{len(bad)} mismatches"

    # 3) user_id coverage: same set across folds and equal to train users
    train_uids = set(pl.read_parquet("data/train.parquet", columns=["user_id"])["user_id"].to_list())
    for name in ["fold_00", "fold_01", "fold_02", "fold_03", "fold_end"]:
        uids = set(read_fold(name)["user_id"].to_list())
        assert uids == train_uids, f"{name} user mismatch"
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
