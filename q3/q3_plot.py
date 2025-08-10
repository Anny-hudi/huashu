import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def detect_base_stations(df: pd.DataFrame):
    bs_list = []
    for col in df.columns:
        if "_URLLC_RB" in col:
            bs = col.split("_URLLC_RB")[0]
            bs_list.append(bs)
    return sorted(list(set(bs_list)))


def plot_reward(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 3))
    plt.plot(df.index, df["reward"], marker='o', linewidth=1)
    plt.title("Reward over decisions")
    plt.xlabel("Decision index")
    plt.ylabel("Reward (avg QoS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_over_time.png"), dpi=150)
    plt.close()


def plot_rb_per_bs(df: pd.DataFrame, bs: str, outdir: str):
    u = df[f"{bs}_URLLC_RB"]
    e = df[f"{bs}_eMBB_RB"]
    m = df[f"{bs}_mMTC_RB"]

    plt.figure(figsize=(8, 3))
    plt.stackplot(df.index, u, e, m, labels=["URLLC", "eMBB", "mMTC"], alpha=0.8)
    plt.title(f"{bs} RB allocation")
    plt.xlabel("Decision index")
    plt.ylabel("RBs")
    plt.legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{bs}_rb_allocation.png"), dpi=150)
    plt.close()


def plot_power_per_bs(df: pd.DataFrame, bs: str, outdir: str):
    pu = df[f"{bs}_URLLC_P"]
    pe = df[f"{bs}_eMBB_P"]
    pm = df[f"{bs}_mMTC_P"]

    plt.figure(figsize=(8, 3))
    plt.plot(df.index, pu, label="URLLC", marker='o', linewidth=1)
    plt.plot(df.index, pe, label="eMBB", marker='o', linewidth=1)
    plt.plot(df.index, pm, label="mMTC", marker='o', linewidth=1)
    plt.title(f"{bs} power levels")
    plt.xlabel("Decision index")
    plt.ylabel("Power (dBm)")
    plt.ylim(0, 40)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{bs}_power_levels.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="q3_eval_decisions.csv", help="evaluation decisions csv path")
    parser.add_argument("--outdir", type=str, default="q3_plots", help="output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    ensure_dir(args.outdir)
    df = pd.read_csv(args.csv)

    # Plot reward
    if "reward" in df.columns:
        plot_reward(df, args.outdir)

    # Detect BS and plot per BS
    bs_list = detect_base_stations(df)
    for bs in bs_list:
        plot_rb_per_bs(df, bs, args.outdir)
        plot_power_per_bs(df, bs, args.outdir)

    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()

