"""EDA of data/train.parquet: integrity, target structure, persistence,
seasonality/calendar, intent signals, segments. Prints findings and writes
reports/eda_2026_08_24.md."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

TRAIN_PATH = "data/train.parquet"
REPORT_PATH = "reports/eda_2026_08_24.md"

CV_ANCHORS = {
    "fold_00": date(2025, 12, 3),
    "fold_01": date(2025, 12, 17),
    "fold_02": date(2025, 12, 31),
    "fold_03": date(2026, 1, 14),
}
FOLD_END_ANCHOR = date(2026, 2, 13)

LINES: list[str] = []


def out(text: str = "") -> None:
    print(text)
    LINES.append(text)


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    z_t, z_p = np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((z_t - z_p) ** 2)))


def section(title: str) -> None:
    out(f"\n## {title}\n")


def main() -> None:
    lf = pl.scan_parquet(TRAIN_PATH)

    section("1. Схема и целостность")
    schema = lf.collect_schema()
    out(f"колонки ({len(schema)}): {schema.names()}")
    n_rows = lf.select(pl.len()).collect().item()
    stats = collect(
        lf.select(
            pl.col("user_id").n_unique().alias("users"),
            pl.col("event_date").min().alias("d_min"),
            pl.col("event_date").max().alias("d_max"),
        )
    ).row(0, named=True)
    out(f"строк: {n_rows:,}, юзеров: {stats['users']:,}, даты: {stats['d_min']} .. {stats['d_max']}")
    cal_days = (stats["d_max"] - stats["d_min"]).days + 1
    out(f"календарных дней в диапазоне: {cal_days}, плотность рядов: {n_rows / (stats['users'] * cal_days):.1%}")

    nulls = collect(lf.select([pl.col(c).null_count().alias(c) for c in schema.names()]))
    n_null_cols = sum(nulls.row(0, named=True)[c] > 0 for c in schema.names())
    out(f"пропуски: {n_null_cols} колонок с NaN")

    dup = collect(
        lf.group_by("user_id", "event_date").agg(pl.len().alias("n")).filter(pl.col("n") > 1).select(pl.len())
    ).item()
    out(f"дубликаты (user_id, event_date): {dup}")

    empty_rows = collect(
        lf.filter(
            (pl.col("searches") == 0) & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0) & (pl.col("gmv") == 0)
        )
        .select(pl.len())
    ).item()
    out(f"полностью пустых дней в данных: {empty_rows} ({empty_rows / n_rows:.2%}) — прореженные ряды подтверждены" )

    num_cols = [c for c in schema.names() if c not in ("event_date",)]
    mins = collect(lf.select([pl.col(c).min().alias(c) for c in num_cols]).head(1)).row(0, named=True)
    neg = {c: v for c, v in mins.items() if isinstance(v, (int, float)) and v < 0}
    out(f"отрицательные значения минимумов: {neg if neg else 'нет'}")

    cons = collect(
        lf.select(
            (pl.col("gmv") != pl.col("gmv_search") + pl.col("gmv_cat")).sum().alias("gmv_split"),
            (pl.col("to_ord") != pl.col("search_to_ord") + pl.col("cat_to_ord")).sum().alias("ord_split"),
            (pl.col("to_cart") != pl.col("search_to_cart") + pl.col("cat_to_cart")).sum().alias("cart_split"),
            (pl.col("has_search_to_ord") != (pl.col("search_to_ord") > 0)).sum().alias("flag_ord"),
            ((pl.col("gmv") > 0) & (pl.col("to_ord") == 0)).sum().alias("gmv_wo_ord"),
            ((pl.col("to_ord") > 0) & (pl.col("gmv") <= 0)).sum().alias("ord_wo_gmv"),
        )
    ).row(0, named=True)
    out(f"несоответствия тождеств: {cons}")

    section("2. Календарь: тренд и сезонность платформы")
    daily = collect(
        lf.group_by("event_date")
        .agg(
            pl.col("gmv").sum().alias("gmv"),
            pl.col("to_ord").sum().alias("ords"),
            pl.col("searches").sum().alias("searches"),
            pl.col("user_id").n_unique().alias("active_users"),
        )
        .sort("event_date")
    )
    daily = daily.with_columns(pl.col("event_date").dt.weekday().alias("dow"))
    dow = daily.group_by("dow").agg(pl.col("gmv").mean().alias("gmv_mean")).sort("dow")
    gm = dow["gmv_mean"].to_numpy()
    out("средний GMV платформы по дням недели (1=Пн): " + ", ".join(f"{i+1}:{v/1e6:.1f}M" for i, v in enumerate(gm)))
    out(f"разброс DoW: min/max = {gm.min()/gm.max():.2f}")

    monthly = daily.with_columns(pl.col("event_date").dt.month_start().alias("m")).group_by("m").agg(
        pl.col("gmv").sum().alias("gmv"),
        pl.col("active_users").mean().alias("avg_active_users"),
        pl.col("gmv").mean().alias("gmv_per_day"),
    ).sort("m")
    out("месячные агрегаты (GMV/млн, средние дневные активные юзеры/тыс, GMV/день на активного юзера):")
    for r in monthly.iter_rows(named=True):
        gpu = r["gmv_per_day"] / max(r["avg_active_users"], 1)
        out(
            f"  {r['m']:%Y-%m}: gmv={r['gmv']/1e6:7.1f}M  act_users/day={r['avg_active_users']/1e3:5.1f}k  "
            f"gmv/user/day={gpu:6.0f}"
        )

    def win_sum(d0: date, d1: date) -> float:
        return float(daily.filter(pl.col("event_date").is_between(d0, d1))["gmv"].sum())

    twin = win_sum(date(2025, 2, 14), date(2025, 3, 15))
    pre = win_sum(date(2025, 1, 15), date(2025, 2, 13))
    post = win_sum(date(2025, 3, 16), date(2025, 4, 14))
    out(f"\nблизнец тестового окна [14.02–15.03.25]: {twin/1e6:.1f}M")
    out(f"смежное до [15.01–13.02.25]: {pre/1e6:.1f}M (index {twin/pre:.3f})")
    out(f"смежное после [16.03–14.04.25]: {post/1e6:.1f}M (index {twin/post:.3f})")
    cur_pre = win_sum(date(2026, 1, 15), date(2026, 2, 13))
    out(f"окно перед сабмитом [15.01–13.02.26]: {cur_pre/1e6:.1f}M (YoY к прошлому году: {cur_pre/pre:.3f})")

    twin_daily = daily.filter(pl.col("event_date").is_between(date(2025, 2, 1), date(2025, 3, 31)))
    tbase = float(daily.filter(pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13)))["gmv"].mean())
    out("\nпрофиль близнецового окна 2025 (gmv дня / средняя база января):")
    out("  " + " ".join(
        f"{r['event_date']:%d.%m}:{r['gmv']/tbase:.1f}" for r in twin_daily.iter_rows(named=True)
        if r["event_date"].day in (20, 21, 22, 23, 24, 25) or r["event_date"] in
        (date(2025, 3, 6), date(2025, 3, 7), date(2025, 3, 8), date(2025, 3, 9))
        or r["event_date"].day == 14 or r["event_date"].day == 1
    ))

    section("3. Пользователи: история и активность")
    df_user = collect(
        lf.group_by("user_id")
        .agg(
            pl.len().alias("active_days"),
            pl.col("event_date").min().alias("first_day"),
            pl.col("event_date").max().alias("last_day"),
            pl.col("gmv").sum().alias("gmv_total"),
            pl.col("to_ord").sum().alias("ord_total"),
            pl.col("searches").sum().alias("searches_total"),
            pl.col("to_cart").sum().alias("cart_total"),
            (pl.col("to_ord") > 0).sum().alias("order_days"),
        )
    )
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    ad = np.quantile(df_user["active_days"].to_numpy(), q)
    out(f"активных дней на юзера, квантили 10/25/50/75/90: {np.round(ad, 1)} из {cal_days}")
    od = np.quantile(df_user["order_days"].to_numpy(), q)
    out(f"дней с заказами на юзера, квантили: {np.round(od, 1)}")
    buyers = df_user.filter(pl.col("ord_total") > 0).height
    out(f"юзеров с >=1 заказом за весь период: {buyers:,} ({buyers/stats['users']:.1%})")
    late = df_user.filter(pl.col("first_day") > date(2025, 1, 31)).height
    out(f"юзеров, впервые появившихся после января 2025: {late:,} ({late/stats['users']:.1%})")
    gmv_q = np.quantile(df_user["gmv_total"].to_numpy(), [0.5, 0.9, 0.99])
    top_share = np.sort(df_user["gmv_total"].to_numpy())
    out(f"общий GMV юзера, квантили 50/90/99: {np.round(gmv_q, 0)}; топ-1% юзеров держит "
        f"{top_share[-len(top_share)//100:].sum()/top_share.sum():.1%} всего GMV")

    section("4. Таргет по фолдам: структура, персистентность, наив")
    out("anchor     | zeros | mean z | sd z | p50/p90/p99 y       | naive RMSLE | corr(z_past,z_fut)")
    for name, anchor in CV_ANCHORS.items():
        w0, w1 = anchor + timedelta(days=1), anchor + timedelta(days=30)
        p0, p1 = anchor - timedelta(days=29), anchor
        tgt = collect(
            lf.filter(pl.col("event_date").is_between(w0, w1))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("y"))
        )
        past = collect(
            lf.filter(pl.col("event_date").is_between(p0, p1))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("x"))
        )
        j = tgt.join(past, on="user_id", how="full").fill_null(0.0)
        y = j["y"].to_numpy()
        x = j["x"].to_numpy()
        z = np.log1p(y)
        naive = rmsle(y, x)
        corr = float(np.corrcoef(np.log1p(x), z)[0, 1])
        out(
            f"{name} {anchor}| {np.mean(y == 0):5.1%} | {z.mean():5.2f} | {z.std():4.2f} | "
            f"{np.quantile(y, [0.5, 0.9, 0.99]).round(0)} | {naive:8.5f} | {corr:.3f}"
        )
    out("(в outer join нулевые юзеры = нет активности в окне => y=0)")

    section("5. Сегменты по рекенси на якоре fold_03 (2026-01-14)")
    anchor = CV_ANCHORS["fold_03"]
    w0, w1 = anchor + timedelta(days=1), anchor + timedelta(days=30)
    hist = collect(
        lf.filter(pl.col("event_date") <= anchor)
        .group_by("user_id")
        .agg(
            pl.col("event_date").max().alias("last_any"),
            pl.col("event_date").filter(pl.col("to_ord") > 0).max().alias("last_ord"),
            pl.col("event_date").min().alias("first_day"),
            pl.col("gmv").sum().alias("life_gmv"),
            pl.col("to_ord").sum().alias("life_ord"),
        )
    )
    tgt = collect(
        lf.filter(pl.col("event_date").is_between(w0, w1)).group_by("user_id").agg(pl.col("gmv").sum().alias("y"))
    )
    seg = hist.join(tgt, on="user_id", how="left").with_columns(
        (anchor - pl.col("last_any")).dt.total_days().alias("rec_any"),
        (anchor - pl.col("last_ord")).dt.total_days().alias("rec_ord"),
        pl.col("y").fill_null(0.0),
    )
    bins = [
        ("заказ <7д", pl.col("rec_ord") < 7),
        ("заказ 8–30д", pl.col("rec_ord").is_between(7, 30)),
        ("заказ 31–90д", pl.col("rec_ord").is_between(31, 90)),
        ("заказ 91–180д", pl.col("rec_ord").is_between(91, 180)),
        ("заказ >180д / никогда", (pl.col("rec_ord") > 180) | pl.col("rec_ord").is_null()),
    ]
    out("сегмент                  | юзеров | доля GMV окна | mean y | zeros | naive RMSLE сегмента")
    total_y = seg["y"].sum()
    for label, cond in bins:
        s = seg.filter(cond)
        if s.height == 0:
            out(f"{label:<24} |      0 |           —   |      — |     — |     —")
            continue
        y_s = s["y"].to_numpy()
        past = collect(
            lf.filter(
                pl.col("event_date").is_between(anchor - timedelta(days=29), anchor)
                & pl.col("user_id").is_in(s["user_id"].implode())
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("x"))
        )
        sj = s.join(past, on="user_id", how="left").fill_null(0.0)
        m = rmsle(sj["y"].to_numpy(), sj["x"].to_numpy())
        out(
            f"{label:<24} | {sj.height:>6} | {sj['y'].sum()/total_y:13.1%} | {sj['y'].mean():7.1f} | "
            f"{np.mean(y_s == 0):5.1%} | {m:.3f}"
        )

    section("6. Интент-сигналы: поиски/корзины без заказов -> будущий GMV")
    i0, i1 = anchor - timedelta(days=13), anchor
    intent = collect(
        lf.filter(pl.col("event_date").is_between(i0, i1))
        .group_by("user_id")
        .agg(
            pl.col("searches").sum().alias("s14"),
            pl.col("to_cart").sum().alias("c14"),
            pl.col("to_ord").sum().alias("o14"),
            pl.col("gmv").sum().alias("g14"),
        )
    )
    ij = seg.select("user_id", "y", "rec_ord", "life_ord").join(intent, on="user_id", how="full").fill_null(0)
    nonbuyers = ij.filter(pl.col("o14") == 0)
    b_with_intent = nonbuyers.filter((pl.col("s14") + pl.col("c14")) > 0)
    b_no_intent = nonbuyers.filter((pl.col("s14") + pl.col("c14")) == 0)
    out(f"без заказов за 14д до якоря: {nonbuyers.height:,}")
    out(f"  c поисками/корзиной: {b_with_intent.height:,} -> mean y = {b_with_intent['y'].mean():.1f}, "
        f"P(y>0) = {(b_with_intent['y'] > 0).mean():.1%}")
    out(f"  без активности:      {b_no_intent.height:,} -> mean y = {b_no_intent['y'].mean():.1f}, "
        f"P(y>0) = {(b_no_intent['y'] > 0).mean():.1%}")
    dec = np.quantile(ij["s14"].to_numpy(), np.linspace(0.2, 1.0, 5))
    out("\nP(y>0) и mean y по децилям searches_14d:")
    edges = np.unique(np.r_[0, dec])
    lab = np.digitize(ij["s14"].to_numpy(), edges[1:], right=True)
    for k in range(len(edges)):
        m = lab == k
        if m.sum() > 0:
            ys = ij["y"].to_numpy()[m]
            out(f"  s14<= {edges[k]:>6.0f}: n={m.sum():>6,}  P(y>0)={np.mean(ys > 0):5.1%}  mean y={ys.mean():7.1f}")

    section("7. Хвост и форма z = log1p(y) для лосса")
    y_all = seg["y"].to_numpy()
    z_all = np.log1p(y_all)
    out(f"skew(z) = {float(((z_all - z_all.mean()) ** 3).mean() / z_all.std() ** 3):.2f}, "
        f"kurtosis(z) = {float(((z_all - z_all.mean()) ** 4).mean() / z_all.std() ** 4):.2f}")
    out(f"квантили z: {np.round(np.quantile(z_all, [0.01, 0.25, 0.5, 0.75, 0.99]), 2)}")
    out(f"доля y>10000: {np.mean(y_all > 10000):.2%}, y>50000: {np.mean(y_all > 50000):.3%}")

    section("8. Follow-up: YoY одной когорты, семантика пустых дней, холодный старт")
    jan25 = collect(
        lf.filter(pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13)))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("g25"), pl.len().alias("n25"))
    )
    jan26 = collect(
        lf.filter(pl.col("event_date").is_between(date(2026, 1, 15), date(2026, 2, 13)))
        .group_by("user_id").agg(pl.col("gmv").sum().alias("g26"), pl.len().alias("n26"))
    )
    coh = jan25.join(jan26, on="user_id", how="inner")
    out(f"когорта юзеров, активных в обоих окнах [15.01–13.02]: {coh.height:,} из {jan25.height:,} (2025)")
    out(f"per-user GMV: {coh['g25'].mean():.1f} -> {coh['g26'].mean():.1f} "
        f"(YoY индекс на юзера = {coh['g26'].mean()/coh['g25'].mean():.3f})")
    out(f"per-user активных дней: {coh['n25'].mean():.1f} -> {coh['n26'].mean():.1f}")
    out(f"per-user GMV/активный день: {coh['g25'].sum()/coh['n25'].sum():.1f} -> "
        f"{coh['g26'].sum()/coh['n26'].sum():.1f}")

    flags_empty = collect(
        lf.filter((pl.col("searches") == 0) & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0))
        .select(
            pl.len().alias("n"),
            ((pl.col("search") == 1) | (pl.col("cat") == 1)).mean().alias("with_flag"),
            (pl.col("cat") == 1).mean().alias("cat"),
            (pl.col("search") == 1).mean().alias("search"),
        )
    ).row(0, named=True)
    out(f"\nпустые по конверсии дни: {flags_empty['n']:,}; из них с флагом визита: "
        f"{flags_empty['with_flag']:.1%} (search={flags_empty['search']:.1%}, cat={flags_empty['cat']:.1%})")

    fe_hist = collect(
        lf.filter(pl.col("event_date") <= FOLD_END_ANCHOR)
        .group_by("user_id")
        .agg(pl.col("event_date").max().alias("last_any"),
             pl.col("event_date").filter(pl.col("to_ord") > 0).max().alias("last_ord"))
        .with_columns(
            (FOLD_END_ANCHOR - pl.col("last_any")).dt.total_days().alias("rec_any"),
            (FOLD_END_ANCHOR - pl.col("last_ord")).dt.total_days().alias("rec_ord"),
        )
    )
    for thr in (7, 14, 30, 60):
        n = fe_hist.filter(pl.col("rec_any") >= thr).height
        no = fe_hist.filter(pl.col("rec_ord").is_null() | (pl.col("rec_ord") >= thr)).height
        out(f"fold_end: без активности >= {thr:>2}д: {n:>7,} ({n/2.5e5:.1%}); без заказа >= {thr:>2}д/никогда: {no:>7,} ({no/2.5e5:.1%})")

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# EDA train.parquet — 2026-08-24\n\n"
        "Автоотчёт: `src/eda.py`. Назначение — проверка гипотез о структуре данных "
        "для приоритизации экспериментов 2–5.\n"
    )
    Path(REPORT_PATH).write_text(header + "\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nreport saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
