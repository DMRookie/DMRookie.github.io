"""生成「强化学习」系列四篇博客的配图。

用法：python3 scripts/gen_rl_figures.py
输出：source/images/<文章文件名>/NN_xxx.png
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "PingFang HK", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.grid"] = False

BLUE = "#2f6fb3"
ORANGE = "#e08a2e"
GREEN = "#3f9d5a"
RED = "#c0392b"
GRAY = "#8a949c"
PURPLE = "#7d5ba6"
LIGHT = "#eef3f8"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "source", "images")

P1 = "强化学习-基础介绍"
P2 = "强化学习-价值方法"
P3 = "强化学习-策略梯度"
P4 = "强化学习-基于模型的方法"


def save(fig, post, name):
    d = os.path.join(IMG, post)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path)
    plt.close(fig)
    print("saved", os.path.relpath(path, ROOT))


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=11, weight="normal", tc="black"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            fc=fc, ec=ec, lw=1.6, mutation_aspect=1,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.5)


def arrow(ax, p1, p2, color="#444444", lw=1.6, style="->", rad=0.0, ls="-"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle=style, lw=lw, color=color, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2),
    )


def clean(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# ----------------------------------------------------------------------------
# 第一篇
# ----------------------------------------------------------------------------
def p1_supervised_vs_rl():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    clean(a1, (0, 10), (0, 6.4))
    a1.set_title("监督学习：反馈是指导性的", fontsize=12, fontweight="bold", color=BLUE)
    box(a1, 0.4, 3.6, 2.6, 1.3, "固定数据集\n$(x, y)$，i.i.d.", fc="#e8f0f8")
    box(a1, 4.0, 3.6, 2.2, 1.3, "模型 $f_\\theta$")
    box(a1, 7.2, 3.6, 2.2, 1.3, "预测 $\\hat{y}$")
    box(a1, 4.0, 1.1, 5.4, 1.1, "损失 $\\mathcal{L}(\\hat{y}, y)$ → 梯度直接指出该往哪改",
        fc="#f4ece0", ec=ORANGE)
    arrow(a1, (3.0, 4.25), (4.0, 4.25))
    arrow(a1, (6.2, 4.25), (7.2, 4.25))
    arrow(a1, (8.3, 3.6), (8.3, 2.2), color=ORANGE)
    arrow(a1, (4.6, 2.2), (4.6, 3.6), color=ORANGE)
    a1.text(1.7, 2.6, "标签 $y$ 已知\n数据分布固定", ha="center", va="center",
            fontsize=10.5, color=GRAY, linespacing=1.6)

    clean(a2, (0, 10), (0, 6.4))
    a2.set_title("强化学习：反馈是评价性的", fontsize=12, fontweight="bold", color=RED)
    box(a2, 0.8, 3.4, 3.2, 1.5, "智能体 agent\n$\\pi_\\theta(a \\mid s)$", fc="#e8f0f8")
    box(a2, 6.0, 3.4, 3.2, 1.5, "环境 environment\n$T,\\; R$（未知）", fc="#f0eef8", ec=PURPLE)
    arrow(a2, (4.0, 4.55), (6.0, 4.55), color=BLUE, rad=-0.25)
    a2.text(5.0, 5.5, "动作 $a_t$", ha="center", fontsize=10.5, color=BLUE)
    arrow(a2, (6.0, 3.75), (4.0, 3.75), color=GREEN, rad=-0.25)
    a2.text(5.0, 2.55, "状态 $s_{t+1}$，奖励 $r_t$", ha="center", fontsize=10.5, color=GREEN)
    box(a2, 0.8, 0.7, 8.4, 1.3,
        "只有一个标量分数，且可能延迟；数据由自己产生，分布随策略漂移\n→ 必须探索，误差会随时间复合放大",
        fc="#f8ecea", ec=RED, fs=10.5)
    save(fig, P1, "01_supervised_vs_rl.png")


def p1_discount():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    k = np.arange(0, 101)
    for g, c in zip([0.5, 0.9, 0.99], [RED, BLUE, GREEN]):
        ax.plot(k, g ** k, color=c, lw=2,
                label=f"$\\gamma={g}$（有效视野 $\\approx${int(round(1/(1-g)))} 步）")
        ax.axvline(1 / (1 - g), color=c, ls="--", lw=1.1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("未来第 $k$ 步")
    ax.set_ylabel("该步奖励的权重 $\\gamma^k$")
    ax.set_title("折扣因子决定智能体“看多远”（虚线为 $1/(1-\\gamma)$）", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=10.5)
    save(fig, P1, "02_discount_horizon.png")


def p1_value_iteration():
    data = np.array([
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 1.00],
        [0.00, 0.90, 1.00],
        [0.81, 0.90, 1.00],
        [0.81, 0.90, 1.00],
    ])
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1.05, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=11.5, color="black")
    ax.set_xticks(range(3), ["$s_0$（起点）", "$s_1$", "$s_2$"])
    ax.set_yticks(range(5), [f"$k={i}$" for i in range(5)])
    ax.set_title("价值迭代：奖励从终点一格格传回起点（$\\gamma=0.9$）", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85, label="$V(s)$")
    save(fig, P1, "03_value_iteration.png")


def p1_three_routes():
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    clean(ax, (0, 12), (0, 7.6))
    box(ax, 3.4, 6.1, 5.2, 1.2,
        "强化学习问题：MDP $(\\mathcal{S}, \\mathcal{A}, T, R, \\gamma)$",
        fc="#e8f0f8", fs=12, weight="bold")

    cols = [
        (0.25, BLUE, "① Value-based（第二篇）",
         "学 $Q(s,a)$\n策略 $=\\arg\\max_a Q$\n\nQ-learning → DQN\nDouble / Dueling\nPER / n-step\nC51 → Rainbow"),
        (4.15, GREEN, "② Policy-based（第三篇）",
         "直接优化 $\\pi_\\theta(a \\mid s)$\n对 $J(\\theta)$ 梯度上升\n\nREINFORCE → A2C\nTRPO → PPO\nDDPG → TD3 → SAC\nRLHF / GRPO"),
        (8.05, PURPLE, "③ Model-based（第四篇）",
         "学 $\\hat{T}, \\hat{R}$\n在模型里规划/想象\n\nDyna → MBPO\nPETS / CEM+MPC\nWorld Models → Dreamer\nMCTS → AlphaZero → MuZero"),
    ]
    for x, c, title, body in cols:
        box(ax, x, 3.75, 3.7, 0.85, title, fc="white", ec=c, fs=11.5, weight="bold", tc=c)
        box(ax, x, 0.5, 3.7, 3.0, body, fc="#fafbfc", ec=c, fs=10.5)
        arrow(ax, (6.0, 6.1), (x + 1.85, 4.6), color=c, lw=1.5, rad=0.08)
    ax.text(6.0, 7.45, "“我们究竟要学什么？”决定了三条技术路线",
            ha="center", fontsize=11, color=GRAY)
    save(fig, P1, "04_three_routes.png")


def p1_deadly_triad():
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    clean(ax, (-3.6, 3.6), (-3.2, 3.6))
    ax.set_aspect("equal")
    centers = [(0.0, 1.15), (-1.05, -0.75), (1.05, -0.75)]
    colors = [BLUE, GREEN, ORANGE]
    labels = [
        ("函数逼近\nfunction approximation", (0.0, 3.1)),
        ("自举\nbootstrapping", (-2.55, -2.35)),
        ("off-policy\n训练", (2.5, -2.35)),
    ]
    for (cx, cy), c in zip(centers, colors):
        ax.add_patch(Circle((cx, cy), 1.75, fc=c, ec=c, alpha=0.22, lw=2))
    for (txt, pos), c in zip(labels, colors):
        ax.text(pos[0], pos[1], txt, ha="center", va="center",
                fontsize=11.5, color=c, fontweight="bold", linespacing=1.5)
    ax.text(0, -0.15, "训练\n可能发散", ha="center", va="center",
            fontsize=12.5, color=RED, fontweight="bold", linespacing=1.5)
    ax.set_title("致命三角：任意两个凑一起通常没事，三个齐了就危险", fontsize=12)
    save(fig, P1, "05_deadly_triad.png")


# ----------------------------------------------------------------------------
# 第二篇
# ----------------------------------------------------------------------------
def p2_nstep_tradeoff():
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    n = np.arange(1, 13)
    bias = 1.0 * 0.65 ** (n - 1)
    var = 0.10 * n ** 1.25
    total = bias + var
    ax.plot(n, bias, "o-", color=BLUE, lw=2, label="自举偏差（$n$ 越大越小）")
    ax.plot(n, var, "s-", color=ORANGE, lw=2, label="回报方差（$n$ 越大越大）")
    ax.plot(n, total, "^-", color=RED, lw=2.4, label="总误差")
    best = n[np.argmin(total)]
    ax.axvline(best, color=GRAY, ls="--", lw=1.2)
    ax.annotate(f"实践中的甜点区\n$n \\approx 3\\sim5$", xy=(best, total.min()),
                xytext=(best + 1.6, total.min() + 0.75), fontsize=10.5, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    ax.set_xlabel("多步回报的步数 $n$")
    ax.set_ylabel("误差（示意，单位任意）")
    ax.set_title("$n$-step return：偏差-方差的旋钮（$n=1$ 为 TD，$n\\to\\infty$ 为 MC）", fontsize=11.5)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10.5)
    save(fig, P2, "01_nstep_tradeoff.png")


def p2_cliff_walking():
    grid = np.zeros((4, 12))
    grid[3, 1:11] = 1.0
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#f7f9fb", "#4a4a4a"]),
              vmin=0, vmax=1)
    for x in range(13):
        ax.axvline(x - 0.5, color="#c9d2da", lw=1)
    for y in range(5):
        ax.axhline(y - 0.5, color="#c9d2da", lw=1)
    ax.text(5.5, 3, "悬崖：奖励 $-100$，回到起点", ha="center", va="center",
            fontsize=11, color="white")

    q_path = [(0, 3), (0, 2)] + [(x, 2) for x in range(1, 12)] + [(11, 3)]
    s_path = [(0, 3), (0, 2), (0, 1), (0, 0)] + [(x, 0) for x in range(1, 12)] + \
             [(11, 1), (11, 2), (11, 3)]
    ax.plot([p[0] - 0.07 for p in q_path], [p[1] - 0.07 for p in q_path], "-o", color=RED,
            lw=2.4, ms=5, zorder=3,
            label="Q-learning（off-policy）：贴着悬崖，最短但训练时常掉下去")
    ax.plot([p[0] + 0.07 for p in s_path], [p[1] + 0.07 for p in s_path], "-s", color=BLUE,
            lw=2.4, ms=5, zorder=3,
            label="SARSA（on-policy）：绕远，训练期间更安全")
    halo = [pe.withStroke(linewidth=3.5, foreground="white")]
    ax.text(-0.02, 3.32, "S", ha="center", va="center", fontsize=15, fontweight="bold",
            color=BLUE, zorder=6, path_effects=halo)
    ax.text(11.02, 3.34, "G", ha="center", va="center", fontsize=15, fontweight="bold",
            color=GREEN, zorder=6, path_effects=halo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("悬崖行走：同一个环境，两种算法学出不同的路", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)
    save(fig, P2, "02_cliff_walking.png")


def p2_dqn_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    clean(ax, (0, 12.4), (0, 6.6))
    box(ax, 0.2, 4.4, 2.1, 1.3, "环境\nenvironment", fc="#f0eef8", ec=PURPLE)
    box(ax, 3.1, 4.4, 3.0, 1.3, "经验回放池 $\\mathcal{D}$\n（$10^5\\sim10^6$ 条）", fc="#e8f0f8")
    box(ax, 7.0, 4.4, 2.4, 1.3, "随机采样\nmini-batch", fc="#eef7f0", ec=GREEN)
    box(ax, 4.6, 2.15, 2.6, 1.2, "在线网络 $Q(\\cdot;\\theta)$", fc="white", ec=BLUE)
    box(ax, 8.6, 2.15, 2.8, 1.2, "目标网络 $Q(\\cdot;\\theta^{-})$", fc="white", ec=ORANGE)
    box(ax, 4.2, 0.15, 7.6, 1.15,
        "$\\mathcal{L} = ( r + \\gamma \\max_{a'} Q(s',a';\\theta^{-}) - Q(s,a;\\theta) )^2$（Huber）",
        fc="#f8ecea", ec=RED, fs=11.5)

    arrow(ax, (2.3, 5.05), (3.1, 5.05))
    ax.text(2.7, 5.45, "$(s,a,r,s')$", ha="center", fontsize=10)
    arrow(ax, (6.1, 5.05), (7.0, 5.05))
    arrow(ax, (8.0, 4.4), (6.2, 3.35), color=GREEN, rad=0.15)
    arrow(ax, (8.6, 4.4), (10.0, 3.35), color=GREEN, rad=-0.15)
    arrow(ax, (5.9, 2.15), (6.6, 1.3), color=BLUE)
    arrow(ax, (10.0, 2.15), (9.4, 1.3), color=ORANGE)
    arrow(ax, (4.2, 0.7), (1.25, 4.4), color=BLUE, rad=0.22)
    ax.text(1.15, 2.6, "$\\epsilon$-greedy\n采取动作", ha="center", fontsize=10, color=BLUE)
    arrow(ax, (7.2, 2.75), (8.6, 2.75), color=ORANGE, ls="--")
    ax.text(7.9, 3.15, "每 $C$ 步同步", ha="center", fontsize=9.8, color=ORANGE)
    ax.set_title("DQN 全貌：经验回放打破数据相关性，目标网络稳定住标签", fontsize=12)
    save(fig, P2, "03_dqn_pipeline.png")


def p2_overestimation():
    rng = np.random.default_rng(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.0))

    est = rng.normal(0, 1, size=(200000, 5))
    mx = est.max(axis=1)
    a1.hist(est[:, 0], bins=80, color=GRAY, alpha=0.55, density=True,
            label="单个动作的估计 $\\hat{Q}$（无偏）")
    a1.hist(mx, bins=80, color=RED, alpha=0.65, density=True,
            label="$\\max_a \\hat{Q}$（5 个动作）")
    a1.axvline(0, color="black", lw=1.4, ls="--")
    a1.axvline(mx.mean(), color=RED, lw=1.6)
    a1.annotate(f"过估计偏差\n$\\approx {mx.mean():.2f}$", xy=(mx.mean(), 0.28),
                xytext=(1.9, 0.42), fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    a1.text(-4.8, 0.115, "真实值：所有动作均为 0", fontsize=9.8, color="black")
    a1.set_xlabel("估计值")
    a1.set_ylabel("概率密度")
    a1.set_title("零均值噪声经过 $\\max$ 后变成正偏差", fontsize=11.5)
    a1.legend(frameon=False, fontsize=9.8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    a1.grid(alpha=0.2)

    ks = np.arange(2, 21)
    means = [rng.normal(0, 1, size=(60000, k)).max(axis=1).mean() for k in ks]
    a2.plot(ks, means, "o-", color=RED, lw=2)
    a2.axhline(0, color="black", ls="--", lw=1.2)
    a2.set_xlabel("动作数 $|\\mathcal{A}|$")
    a2.set_ylabel("$\\mathbb{E}[\\max_a \\hat{Q}]$")
    a2.set_title("动作越多，$\\max$ 的高估越严重（$\\sigma=1$）", fontsize=11.5)
    a2.grid(alpha=0.25)
    save(fig, P2, "04_overestimation.png")


def p2_rainbow_ablation():
    names = ["Double", "Dueling", "Noisy Nets", "Distributional", "Multi-step", "Prioritized Replay"]
    vals = [0.10, 0.16, 0.35, 0.62, 0.86, 0.92]
    colors = [GRAY, GRAY, ORANGE, ORANGE, RED, RED]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.barh(names, vals, color=colors, alpha=0.85, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + 0.02, y, ["可忽略", "较小", "中等", "较大", "很大", "很大"][y],
                va="center", fontsize=10, color="#555555")
    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    ax.set_xlabel("移除该组件后 Rainbow 的性能下降（示意，相对趋势）")
    ax.set_title("Rainbow 消融：优先回放与多步回报最关键", fontsize=12)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    save(fig, P2, "05_rainbow_ablation.png")


# ----------------------------------------------------------------------------
# 第三篇
# ----------------------------------------------------------------------------
def p3_pg_intuition():
    adv = np.array([2.0, -1.0, 0.5, -1.5, 0.0])
    logits = np.zeros(5)
    p0 = np.exp(logits) / np.exp(logits).sum()
    p1 = np.exp(logits + 0.55 * adv) / np.exp(logits + 0.55 * adv).sum()

    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x - 0.2, p0, width=0.38, color=GRAY, alpha=0.8, label="更新前 $\\pi_\\theta$")
    ax.bar(x + 0.2, p1, width=0.38, color=BLUE, alpha=0.9, label="更新后 $\\pi_{\\theta'}$")
    for i, a in enumerate(adv):
        c = GREEN if a > 0 else (RED if a < 0 else GRAY)
        ax.text(i, max(p0[i], p1[i]) + 0.025, f"$\\hat{{A}}={a:+.1f}$",
                ha="center", fontsize=10.5, color=c)
    ax.set_xticks(x, [f"$a_{i+1}$" for i in range(5)])
    ax.set_ylabel("动作概率")
    ax.set_ylim(0, 0.62)
    ax.set_title("策略梯度 = 用优势加权的最大似然：好动作抬概率，坏动作压概率", fontsize=11.5)
    ax.legend(frameon=False, fontsize=10.5)
    ax.grid(alpha=0.22, axis="y")
    save(fig, P3, "01_pg_intuition.png")


def p3_baseline_variance():
    rng = np.random.default_rng(7)
    mu = np.array([11.0, 10.0, 9.0])
    pi = np.ones(3) / 3
    n_batch, n_trial = 16, 6000
    g_no, g_with = np.empty(n_trial), np.empty(n_trial)
    for i in range(n_trial):
        a = rng.choice(3, size=n_batch, p=pi)
        G = mu[a] + rng.normal(0, 1.0, size=n_batch)
        score = (a == 0).astype(float) - pi[0]      # ∇_θ0 log π(a)
        g_no[i] = np.mean(score * G)
        g_with[i] = np.mean(score * (G - G.mean()))  # 减去 batch 均值作为基线

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bins = np.linspace(-2.2, 2.2, 90)
    ax.hist(g_no, bins=bins, color=RED, alpha=0.55, density=True,
            label=f"无基线：标准差 = {g_no.std():.3f}")
    ax.hist(g_with, bins=bins, color=BLUE, alpha=0.7, density=True,
            label=f"减去基线：标准差 = {g_with.std():.3f}")
    ax.axvline(g_no.mean(), color=RED, lw=1.4, ls="--")
    ax.axvline(g_with.mean(), color=BLUE, lw=1.4, ls="--")
    ax.set_xlabel("梯度估计 $\\hat{g}$（对 $\\theta_0$ 的分量）")
    ax.set_ylabel("概率密度")
    ax.set_title(f"基线只降方差、不引入偏差（两者均值 {g_no.mean():.3f} vs {g_with.mean():.3f}）",
                 fontsize=11.5)
    ax.legend(frameon=False, fontsize=10.5)
    ax.grid(alpha=0.22)
    save(fig, P3, "02_baseline_variance.png")


def p3_gae():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    l = np.arange(0, 61)
    gamma = 0.99
    for lam, c in zip([0.0, 0.5, 0.9, 0.95, 1.0], [RED, ORANGE, GREEN, BLUE, PURPLE]):
        w = (gamma * lam) ** l if lam > 0 else np.where(l == 0, 1.0, 0.0)
        a1.plot(l, np.maximum(w, 1e-6), lw=2, color=c, label=f"$\\lambda={lam}$")
    a1.set_yscale("log")
    a1.set_ylim(1e-5, 2)
    a1.set_xlabel("向后第 $l$ 步的 TD error $\\delta_{t+l}$")
    a1.set_ylabel("权重 $(\\gamma\\lambda)^l$")
    a1.set_title("GAE 把各步 TD error 指数加权求和", fontsize=11.5)
    a1.legend(frameon=False, fontsize=10)
    a1.grid(alpha=0.25, which="both")

    lam = np.linspace(0, 1, 400)
    bias = 1.0 - lam ** 1.2
    var = 0.10 + 1.6 * lam ** 60
    a2.plot(lam, bias, color=BLUE, lw=2, label="偏差（来自自举）")
    a2.plot(lam, var, color=ORANGE, lw=2, label="方差（来自真实回报）")
    a2.plot(lam, bias + var, color=RED, lw=2.4, label="总误差")
    best = lam[np.argmin(bias + var)]
    a2.axvline(best, color=GRAY, ls="--", lw=1.2)
    a2.annotate(f"甜点区 $\\lambda \\approx {best:.2f}$\n（实践常用 0.95）",
                xy=(best, (bias + var).min()), xytext=(0.10, 0.30), fontsize=10.5, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    a2.set_xlabel("$\\lambda$（0 = 一步 TD，1 = 蒙特卡洛）")
    a2.set_ylabel("误差（示意）")
    a2.set_title("$\\lambda$ 是偏差-方差的连续旋钮", fontsize=11.5)
    a2.legend(frameon=False, fontsize=10)
    a2.grid(alpha=0.25)
    save(fig, P3, "03_gae_lambda.png")


def p3_ppo_clip():
    eps = 0.2
    r = np.linspace(0, 2.2, 600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.0), sharex=True)

    for ax, A, title in [(a1, 1.0, "好动作 $\\hat{A} > 0$"), (a2, -1.0, "坏动作 $\\hat{A} < 0$")]:
        unc = r * A
        cl = np.clip(r, 1 - eps, 1 + eps) * A
        L = np.minimum(unc, cl)
        ax.plot(r, unc, color=GRAY, lw=1.4, ls="--", label="$r_t \\hat{A}$（未截断）")
        ax.plot(r, L, color=BLUE if A > 0 else RED, lw=2.6, label="$L^{\\mathrm{CLIP}}$")
        ax.axvspan(1 - eps, 1 + eps, color=GREEN, alpha=0.10)
        ax.axvline(1.0, color="black", lw=1.0, ls=":")
        ax.set_xlabel("重要性比率 $r_t(\\theta) = \\pi_\\theta / \\pi_{\\theta_{old}}$")
        ax.set_title(title, fontsize=11.5)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=10, loc="lower right" if A > 0 else "lower left")
        ax.set_xticks([0, 1 - eps, 1, 1 + eps, 2], ["0", "$1-\\epsilon$", "1", "$1+\\epsilon$", "2"])

    a1.set_ylabel("目标函数值")
    a1.annotate("超过 $1+\\epsilon$ 后变平\n梯度为 0，不再往前推",
                xy=(1.7, 1.2), xytext=(1.05, 0.35), fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    a2.annotate("低于 $1-\\epsilon$ 后变平\n已压够，不再继续压",
                xy=(0.45, -0.8), xytext=(0.06, -1.75), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    a2.annotate("这一侧不设限\n坏动作可以大力压",
                xy=(2.0, -2.0), xytext=(1.15, -1.4), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    fig.suptitle("PPO 的 clip：把“信任域”做成一个梯度开关", fontsize=12.5, y=1.02)
    save(fig, P3, "04_ppo_clip.png")


def p3_taxonomy():
    algos = [
        ("REINFORCE", 0.08, 0.93, GREEN),
        ("A2C / A3C", 0.16, 0.86, GREEN),
        ("TRPO", 0.10, 0.72, GREEN),
        ("PPO", 0.30, 0.80, GREEN),
        ("SAC", 0.84, 0.74, PURPLE),
        ("DDPG", 0.80, 0.12, ORANGE),
        ("TD3", 0.91, 0.20, ORANGE),
        ("DQN / Rainbow", 0.84, 0.38, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvline(0.5, color="#c9d2da", lw=1.4)
    ax.axhline(0.5, color="#c9d2da", lw=1.4)
    for name, x, y, c in algos:
        ax.scatter([x], [y], s=190, color=c, alpha=0.85, zorder=3)
        ax.text(x, y + 0.055, name, ha="center", fontsize=10.5, color=c, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.set_xticks([0.15, 0.85], ["on-policy\n（数据用完即弃）", "off-policy\n（可复用回放池）"])
    ax.set_yticks([0.12, 0.88], ["确定性策略\n$a=\\mu_\\theta(s)$", "随机策略\n$\\pi_\\theta(a \\mid s)$"])
    ax.tick_params(length=0)
    ax.set_title("比“value 还是 policy”更有用的两个维度", fontsize=12)
    for s in ax.spines.values():
        s.set_color("#dde3e9")
    save(fig, P3, "05_algo_taxonomy.png")


# ----------------------------------------------------------------------------
# 第四篇
# ----------------------------------------------------------------------------
def p4_sample_efficiency():
    names = ["PPO\n(model-free, on-policy)", "SAC\n(model-free, off-policy)",
             "MBPO / Dreamer\n(model-based)"]
    vals = [3e6, 3e5, 3e4]
    colors = [GRAY, BLUE, PURPLE]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.barh(names, vals, color=colors, alpha=0.85, height=0.55)
    for y, v in enumerate(vals):
        ax.text(v * 1.2, y, f"$\\sim 10^{{{int(round(np.log10(v)))}}}$ 量级",
                va="center", fontsize=10.5, color="#555555")
    ax.set_xscale("log")
    ax.set_xlim(3e3, 3e7)
    ax.set_xlabel("达到相近性能所需的真实交互步数（连续控制基准，示意）")
    ax.set_title("model-based 的核心卖点：样本效率", fontsize=12)
    ax.grid(alpha=0.22, axis="x", which="both")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    save(fig, P4, "01_sample_efficiency.png")


def p4_mcts():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    clean(ax, (0, 10), (0, 6.6))
    nodes = {
        "root": (5.0, 5.6), "A": (2.6, 4.1), "B": (5.0, 4.1), "C": (7.4, 4.1),
        "B1": (4.0, 2.6), "B2": (6.1, 2.6), "new": (6.1, 1.2),
    }
    edges = [("root", "A"), ("root", "B"), ("root", "C"), ("B", "B1"), ("B", "B2")]
    path = [("root", "B"), ("B", "B2")]
    for u, v in edges:
        c, lw = ("#c2ccd4", 1.6)
        if (u, v) in path:
            c, lw = ORANGE, 3.0
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color=c, lw=lw, zorder=1)
    ax.plot([nodes["B2"][0], nodes["new"][0]], [nodes["B2"][1], nodes["new"][1]],
            color=GREEN, lw=2.2, ls="--", zorder=1)
    for k, (x, y) in nodes.items():
        if k == "new":
            ax.add_patch(Circle((x, y), 0.32, fc="white", ec=GREEN, lw=2.2, ls="--", zorder=2))
            ax.text(x, y, "新", ha="center", va="center", fontsize=10.5, color=GREEN, zorder=3)
        else:
            fc = "#fdf1e0" if k in ("root", "B", "B2") else "#f2f6fa"
            ec = ORANGE if k in ("root", "B", "B2") else BLUE
            ax.add_patch(Circle((x, y), 0.34, fc=fc, ec=ec, lw=2, zorder=2))
            ax.text(x, y, "$N,Q$", ha="center", va="center", fontsize=9, zorder=3)
    ax.text(5.0, 6.25, "当前局面（根节点）", ha="center", fontsize=11, color="#444444")
    box(ax, 7.1, 0.7, 2.7, 1.0, "价值网络 $v_\\theta$ 打分\n（或随机模拟到底）",
        fc="#eef7f0", ec=GREEN, fs=10)
    arrow(ax, (7.1, 1.2), (6.5, 1.2), color=GREEN)
    arrow(ax, (5.75, 1.5), (4.75, 3.85), color=PURPLE, rad=0.3, lw=2)
    ax.text(2.2, 1.55, "④ 回传 backup\n更新路径上所有 $N, Q$",
            ha="center", fontsize=10, color=PURPLE, linespacing=1.5)
    ax.text(1.0, 5.3, "① 选择 selection\n按 PUCT 沿树下行", fontsize=10.5,
            color=ORANGE, linespacing=1.5)
    ax.text(7.6, 2.6, "② 扩展 expansion", fontsize=10.5, color=GREEN)
    ax.text(8.0, 2.05, "③ 模拟 simulation", fontsize=10.5, color=GREEN)
    ax.set_title("MCTS 的四步循环：策略网络剪宽度，价值网络截深度", fontsize=12)
    save(fig, P4, "02_mcts.png")


def p4_compounding_error():
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    h = np.arange(0, 21)
    L = 1.6
    for eps, c in zip([0.01, 0.02, 0.05], [GREEN, BLUE, RED]):
        err = eps * (L ** h - 1) / (L - 1)
        ax.plot(h, err, "o-", ms=3.5, color=c, lw=2, label=f"单步误差 $\\epsilon={eps}$")
    ax.axvspan(0, 5, color=GREEN, alpha=0.10)
    ax.text(2.2, 3e2, "短 rollout\n安全区", ha="center", fontsize=10.5, color=GREEN,
            linespacing=1.5)
    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.set_xlabel("在模型里想象的步数 $h$")
    ax.set_ylabel("累积状态预测误差（示意）")
    ax.set_title("复合误差：一步预测的小误差会指数放大", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=10.5)
    save(fig, P4, "03_compounding_error.png")


def p4_dyna_loop():
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    clean(ax, (0, 11), (0, 5.6))
    box(ax, 0.3, 3.5, 2.5, 1.2, "真实环境\n交互一步", fc="#f0eef8", ec=PURPLE)
    box(ax, 3.6, 3.5, 2.6, 1.2, "真实数据池\n$\\mathcal{D}_{real}$", fc="#e8f0f8")
    box(ax, 7.4, 3.5, 3.2, 1.2, "学模型 $\\hat{T}, \\hat{R}$\n（概率集成）", fc="#fdf1e0", ec=ORANGE)
    box(ax, 7.4, 1.1, 3.2, 1.2, "想象数据\n（从真实状态出发\n只走 1~5 步）", fc="#eef7f0", ec=GREEN, fs=10)
    box(ax, 3.2, 1.1, 3.4, 1.2, "model-free 更新\n（SAC / DQN）", fc="white", ec=BLUE)

    arrow(ax, (2.8, 4.1), (3.6, 4.1))
    arrow(ax, (6.2, 4.1), (7.4, 4.1))
    arrow(ax, (9.0, 3.5), (9.0, 2.3), color=ORANGE)
    arrow(ax, (7.4, 1.7), (6.6, 1.7), color=GREEN, lw=2)
    arrow(ax, (4.9, 3.5), (4.9, 2.3), color=BLUE)
    ax.text(5.35, 2.9, "真实经验\n直接学", fontsize=9.5, color=BLUE, linespacing=1.4)
    arrow(ax, (3.2, 1.7), (1.55, 3.5), color=BLUE, rad=0.2)
    ax.text(1.15, 2.2, "用新策略\n继续交互", fontsize=9.8, color=BLUE, linespacing=1.4)
    ax.text(5.5, 0.35, "一次真实交互 → 配上 $K$ 次想象更新，样本效率提升一个量级",
            ha="center", fontsize=10.5, color=GRAY)
    ax.set_title("Dyna / MBPO：把模型当作数据增强器", fontsize=12)
    save(fig, P4, "04_dyna_loop.png")


def p4_radar():
    labels = ["样本效率", "渐近性能", "决策速度", "训练稳定性", "任务迁移"]
    data = {
        "① Value-based": ([3.0, 5.0, 5.0, 3.0, 2.0], BLUE),
        "② Policy gradient": ([2.5, 5.0, 5.0, 4.0, 2.0], GREEN),
        "③ Model-based": ([5.0, 3.5, 2.0, 3.0, 5.0], PURPLE),
    }
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(6.4, 5.6), subplot_kw=dict(polar=True))
    for name, (vals, c) in data.items():
        v = np.array(vals + vals[:1])
        ax.plot(ang, v, color=c, lw=2.2, label=name)
        ax.fill(ang, v, color=c, alpha=0.13)
    ax.set_xticks(ang[:-1], labels, fontsize=11)
    ax.set_yticks([1, 2, 3, 4, 5], ["", "", "", "", ""])
    ax.set_ylim(0, 5.4)
    ax.grid(alpha=0.35)
    ax.set_title("三条路线的取舍（越外越好，示意）", fontsize=12, pad=26)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", bbox_to_anchor=(-0.22, 1.14))
    save(fig, P4, "05_three_routes_radar.png")


if __name__ == "__main__":
    p1_supervised_vs_rl()
    p1_discount()
    p1_value_iteration()
    p1_three_routes()
    p1_deadly_triad()

    p2_nstep_tradeoff()
    p2_cliff_walking()
    p2_dqn_pipeline()
    p2_overestimation()
    p2_rainbow_ablation()

    p3_pg_intuition()
    p3_baseline_variance()
    p3_gae()
    p3_ppo_clip()
    p3_taxonomy()

    p4_sample_efficiency()
    p4_mcts()
    p4_compounding_error()
    p4_dyna_loop()
    p4_radar()
    print("done")
