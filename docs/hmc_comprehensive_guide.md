# ハミルトニアンモンテカルロ法（HMC）完全教育ガイド

## 目次
1. [物理学的直感：なぜハミルトニアンなのか？](#1-物理学的直感なぜハミルトニアンなのか)
2. [HMCの数学的基礎](#2-hmcの数学的基礎)
3. [アルゴリズムの段階的構築](#3-アルゴリズムの段階的構築)
4. [実装から理解するHMC](#4-実装から理解するhmc)
5. [HMC vs 従来手法：性能比較の詳細分析](#5-hmc-vs-従来手法性能比較の詳細分析)
6. [実践的パラメータチューニング](#6-実践的パラメータチューニング)
7. [高度な応用と拡張](#7-高度な応用と拡張)
8. [よくある問題とデバッグ手法](#8-よくある問題とデバッグ手法)

---

## 1. 物理学的直感：なぜハミルトニアンなのか？

### 1.1 従来のMCMC手法の限界

従来のランダムウォーク・メトロポリス・ヘイスティングス法を振り返ってみましょう。この手法は「酔っ払いの歩行」に例えられることがあります。

```python
# 従来のランダムウォークMHの問題点を可視化
import numpy as np
import matplotlib.pyplot as plt

def random_walk_trajectory(steps=1000):
    """酔っ払いの歩行をシミュレート"""
    position = np.array([0.0, 0.0])
    trajectory = [position.copy()]
    
    for _ in range(steps):
        # ランダムな方向に小さなステップ
        step = np.random.normal(0, 0.5, 2)
        position += step
        trajectory.append(position.copy())
    
    return np.array(trajectory)

trajectory = random_walk_trajectory()
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=0.5)
plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='開始')
plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='終了')
plt.title('ランダムウォークの軌跡\n（非効率な探索）')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(trajectory[:200, 0], alpha=0.8, label='X1', linewidth=0.8)
plt.plot(trajectory[:200, 1], alpha=0.8, label='X2', linewidth=0.8)
plt.title('座標の時系列\n（強い自己相関）')
plt.xlabel('ステップ')
plt.ylabel('値')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("ランダムウォークの問題点：")
print("1. 小さなステップサイズ → 探索が遅い")
print("2. ランダムな方向 → 効率的でない動き")
print("3. 強い自己相関 → 独立サンプルが少ない")
```

**問題の本質**：
- 高次元空間では、ランダムな提案の大部分が低確率領域に向かってしまう
- 小さなステップしか許可されないため、探索が極めて非効率
- 確率分布の「地形」を活用できていない

### 1.2 物理学からの着想：ハミルトニアン力学

HMCは古典力学のハミルトニアン形式から着想を得ています。想像してください：

**比喩1：山を転がるボール**
```
従来のMCMC：   目隠しをした人がランダムに歩き回る
HMC：          坂道を転がるボールの物理的な動き
```

**比喩2：惑星の軌道**
```
従来のMCMC：   天体がランダムに飛び回る
HMC：          重力による自然な軌道運動
```

### 1.3 ハミルトニアン力学の基本概念

#### 位置と運動量
```python
def visualize_hamiltonian_concept():
    """ハミルトニアン力学の概念を可視化"""
    t = np.linspace(0, 4*np.pi, 1000)
    
    # 位置（パラメータ）
    q = np.cos(t)
    # 運動量（補助変数）
    p = -np.sin(t)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 位相空間の軌跡
    axes[0, 0].plot(q, p, 'b-', linewidth=2)
    axes[0, 0].plot(q[0], p[0], 'go', markersize=8, label='開始')
    axes[0, 0].plot(q[-1], p[-1], 'ro', markersize=8, label='終了')
    axes[0, 0].set_xlabel('位置 q（パラメータ）')
    axes[0, 0].set_ylabel('運動量 p（補助変数）')
    axes[0, 0].set_title('位相空間での軌跡\n（エネルギー保存による円軌道）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # 時系列
    axes[0, 1].plot(t, q, label='位置 q', linewidth=2)
    axes[0, 1].plot(t, p, label='運動量 p', linewidth=2)
    axes[0, 1].set_xlabel('時間')
    axes[0, 1].set_ylabel('値')
    axes[0, 1].set_title('時間発展')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # エネルギーの時間変化
    energy = 0.5 * (q**2 + p**2)
    axes[1, 0].plot(t, energy, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('時間')
    axes[1, 0].set_ylabel('総エネルギー H(q,p)')
    axes[1, 0].set_title('ハミルトニアン（総エネルギー）\n理想的には一定値')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 確率密度との対応
    q_range = np.linspace(-3, 3, 100)
    prob_density = np.exp(-0.5 * q_range**2)  # 正規分布
    axes[1, 1].plot(q_range, prob_density, 'k-', linewidth=2, label='目標分布')
    axes[1, 1].hist(q[::10], bins=30, density=True, alpha=0.7, 
                   color='blue', label='HMCサンプル')
    axes[1, 1].set_xlabel('位置 q')
    axes[1, 1].set_ylabel('確率密度')
    axes[1, 1].set_title('目標分布の再現')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_hamiltonian_concept()

print("ハミルトニアン力学の特徴：")
print("1. エネルギー保存：系の総エネルギーが保存される")
print("2. 位相空間：(位置, 運動量)の組み合わせで状態を表現")
print("3. 決定論的発展：初期条件が決まれば軌跡が一意に決まる")
print("4. 可逆性：時間を逆向きに進めることが可能")
```

---

## 2. HMCの数学的基礎

### 2.1 ハミルトニアンの定義

**MCMCにおけるハミルトニアン**：
```
H(q, p) = U(q) + K(p)
```

- **q**: 位置変数（実際に興味のあるパラメータ）
- **p**: 運動量変数（補助変数、サンプリング効率を上げるためのもの）
- **U(q)**: ポテンシャルエネルギー = -log π(q)
- **K(p)**: 運動エネルギー = (1/2)p^T M^(-1) p

### 2.2 物理量と統計量の対応関係

```python
def physics_statistics_correspondence():
    """物理学と統計学の対応関係を説明"""
    
    correspondences = {
        "物理学の概念": [
            "位置 q",
            "運動量 p", 
            "ポテンシャルエネルギー U(q)",
            "運動エネルギー K(p)",
            "総エネルギー H(q,p)",
            "温度",
            "力 F = -∇U(q)"
        ],
        "統計学での意味": [
            "パラメータ（推定したい量）",
            "補助変数（サンプリング用）",
            "負の対数事後確率 -log π(q)",
            "補助変数の二次形式",
            "拡張された空間での確率",
            "分布の「広がり」",
            "対数確率の勾配 ∇log π(q)"
        ],
        "MCMCでの役割": [
            "目標のサンプル",
            "探索方向を決める",
            "サンプルの受理確率に影響",
            "ステップサイズを決める",
            "メトロポリス受理の基準",
            "提案分布の調整",
            "効率的な移動方向を指示"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(correspondences)
    print("物理学と統計学の対応関係：")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    print("重要な洞察：")
    print("• 勾配 ∇log π(q) は「確率の坂道」の方向を示す")
    print("• 運動量 p は坂道を「勢いよく登る」役割")
    print("• エネルギー保存により、低確率領域も通過可能")
    print("• 物理法則により自然で効率的な軌跡が生成される")

physics_statistics_correspondence()
```

### 2.3 ハミルトンの運動方程式

```python
def hamilton_equations_demo():
    """ハミルトンの運動方程式のデモンストレーション"""
    
    print("ハミルトンの運動方程式：")
    print("dq/dt = ∂H/∂p = M^(-1) p    （位置の時間微分）")
    print("dp/dt = -∂H/∂q = -∇U(q)     （運動量の時間微分）")
    print()
    
    # 1次元調和振動子の例
    def harmonic_oscillator_exact(t, q0, p0):
        """解析解"""
        q = q0 * np.cos(t) + p0 * np.sin(t)
        p = -q0 * np.sin(t) + p0 * np.cos(t)
        return q, p
    
    def harmonic_oscillator_euler(dt, steps, q0, p0):
        """オイラー法による数値解"""
        q, p = q0, p0
        trajectory = [(q, p)]
        
        for _ in range(steps):
            # オイラー法（不正確）
            q_new = q + dt * p
            p_new = p - dt * q
            q, p = q_new, p_new
            trajectory.append((q, p))
        
        return np.array(trajectory)
    
    def harmonic_oscillator_leapfrog(dt, steps, q0, p0):
        """リープフロッグ法による数値解"""
        q, p = q0, p0
        trajectory = [(q, p)]
        
        for _ in range(steps):
            # リープフロッグ法（シンプレクティック）
            p_half = p - 0.5 * dt * q
            q_new = q + dt * p_half  
            p_new = p_half - 0.5 * dt * q_new
            q, p = q_new, p_new
            trajectory.append((q, p))
        
        return np.array(trajectory)
    
    # 初期条件
    q0, p0 = 1.0, 0.0
    dt = 0.1
    steps = 100
    t_points = np.arange(0, (steps+1)*dt, dt)
    
    # 解析解
    q_exact, p_exact = harmonic_oscillator_exact(t_points, q0, p0)
    
    # 数値解
    euler_traj = harmonic_oscillator_euler(dt, steps, q0, p0)
    leapfrog_traj = harmonic_oscillator_leapfrog(dt, steps, q0, p0)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 位相空間での軌跡
    axes[0, 0].plot(q_exact, p_exact, 'k-', linewidth=3, label='解析解', alpha=0.8)
    axes[0, 0].plot(euler_traj[:, 0], euler_traj[:, 1], 'r--', 
                   linewidth=2, label='オイラー法', alpha=0.7)
    axes[0, 0].plot(leapfrog_traj[:, 0], leapfrog_traj[:, 1], 'b:', 
                   linewidth=2, label='リープフロッグ法', alpha=0.9)
    axes[0, 0].set_xlabel('位置 q')
    axes[0, 0].set_ylabel('運動量 p')
    axes[0, 0].set_title('位相空間での軌跡比較')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # エネルギー保存
    energy_exact = 0.5 * (q_exact**2 + p_exact**2)
    energy_euler = 0.5 * (euler_traj[:, 0]**2 + euler_traj[:, 1]**2)
    energy_leapfrog = 0.5 * (leapfrog_traj[:, 0]**2 + leapfrog_traj[:, 1]**2)
    
    axes[0, 1].plot(t_points, energy_exact, 'k-', linewidth=3, label='解析解')
    axes[0, 1].plot(t_points, energy_euler, 'r--', linewidth=2, label='オイラー法')
    axes[0, 1].plot(t_points, energy_leapfrog, 'b:', linewidth=2, label='リープフロッグ法')
    axes[0, 1].set_xlabel('時間')
    axes[0, 1].set_ylabel('エネルギー')
    axes[0, 1].set_title('エネルギー保存性の比較')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 時系列
    axes[1, 0].plot(t_points, q_exact, 'k-', linewidth=3, label='解析解 q')
    axes[1, 0].plot(t_points, leapfrog_traj[:, 0], 'b:', linewidth=2, label='リープフロッグ q')
    axes[1, 0].set_xlabel('時間')
    axes[1, 0].set_ylabel('位置 q')
    axes[1, 0].set_title('位置の時間発展')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # エネルギー誤差
    axes[1, 1].plot(t_points, np.abs(energy_euler - energy_exact[0]), 
                   'r--', linewidth=2, label='オイラー法')
    axes[1, 1].plot(t_points, np.abs(energy_leapfrog - energy_exact[0]), 
                   'b:', linewidth=2, label='リープフロッグ法')
    axes[1, 1].set_xlabel('時間')
    axes[1, 1].set_ylabel('|エネルギー誤差|')
    axes[1, 1].set_title('エネルギー保存誤差')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"最終エネルギー誤差：")
    print(f"オイラー法:     {abs(energy_euler[-1] - energy_exact[0]):.6f}")
    print(f"リープフロッグ法: {abs(energy_leapfrog[-1] - energy_exact[0]):.6f}")

hamilton_equations_demo()
```

---

## 3. アルゴリズムの段階的構築

### 3.1 Step 1: 基本的な考え方

```python
def hmc_intuition_step1():
    """HMCの基本的な考え方を段階的に説明"""
    
    print("HMC Step 1: 基本的なアイデア")
    print("=" * 50)
    print()
    print("1. 目標分布 π(q) からサンプリングしたい")
    print("2. 補助変数 p（運動量）を導入")
    print("3. 拡張された分布 π(q,p) = π(q)π(p) を考える")
    print("4. π(p) は一般的に標準正規分布 N(0,I)")
    print("5. π(q,p) からサンプリングし、pを捨ててqを得る")
    print()
    
    # 可視化：2次元 -> 4次元拡張の概念
    np.random.seed(42)
    
    # 目標分布（2次元正規分布）
    mu = np.array([0, 0])
    cov = np.array([[1, 0.8], [0.8, 1]])
    
    # 直接サンプリング
    direct_samples = np.random.multivariate_normal(mu, cov, 1000)
    
    # HMC的なサンプリング（概念的）
    q_samples = []
    for _ in range(1000):
        # 運動量をサンプリング
        p = np.random.normal(0, 1, 2)
        # 位置をサンプリング（この段階では直接サンプリング）
        q = np.random.multivariate_normal(mu, cov)
        q_samples.append(q)
    
    hmc_like_samples = np.array(q_samples)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元の2次元分布
    axes[0].scatter(direct_samples[:, 0], direct_samples[:, 1], alpha=0.6, s=10)
    axes[0].set_title('目標分布 π(q)\n（2次元）')
    axes[0].set_xlabel('q1')
    axes[0].set_ylabel('q2')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # 拡張された空間の概念図
    p_sample = np.random.normal(0, 1, 200)
    axes[1].hist(p_sample, bins=30, alpha=0.7, density=True)
    x_range = np.linspace(-3, 3, 100)
    axes[1].plot(x_range, (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_range**2), 
                'r-', linewidth=2, label='π(p) = N(0,1)')
    axes[1].set_title('運動量分布 π(p)\n（補助変数）')
    axes[1].set_xlabel('p')
    axes[1].set_ylabel('密度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 結果の比較
    axes[2].scatter(direct_samples[:, 0], direct_samples[:, 1], 
                   alpha=0.6, s=10, label='直接サンプリング', color='blue')
    axes[2].scatter(hmc_like_samples[:, 0], hmc_like_samples[:, 1], 
                   alpha=0.6, s=10, label='HMC的手法', color='red')
    axes[2].set_title('結果の比較\n（この段階では同じ）')
    axes[2].set_xlabel('q1')
    axes[2].set_ylabel('q2')
    axes[2].legend()
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("重要なポイント：")
    print("• 運動量変数 p は「捨てる」変数")
    print("• 拡張により効率的な探索が可能になる")
    print("• π(q,p) = π(q)π(p) の形に分解")

hmc_intuition_step1()
```

### 3.2 Step 2: 動力学の導入

```python
def hmc_intuition_step2():
    """Step 2: ハミルトン動力学によるサンプリング"""
    
    print("HMC Step 2: 動力学による効率的探索")
    print("=" * 50)
    print()
    print("1. 運動量 p を使って「勢い」をつけた探索")
    print("2. リープフロッグ積分で軌跡を計算")  
    print("3. エネルギー保存により遠くまで移動可能")
    print("4. 勾配情報を活用した賢い移動")
    print()
    
    def leapfrog_integrator(q_initial, p_initial, epsilon, L, grad_U):
        """リープフロッグ積分器"""
        q, p = q_initial.copy(), p_initial.copy()
        trajectory = [(q.copy(), p.copy())]
        
        # 最初の半ステップ
        p = p + 0.5 * epsilon * grad_U(q)
        
        for _ in range(L):
            # 位置の更新
            q = q + epsilon * p
            # 運動量の更新（最後以外）
            if _ < L - 1:
                p = p + epsilon * grad_U(q)
            trajectory.append((q.copy(), p.copy()))
        
        # 最後の半ステップ
        p = p + 0.5 * epsilon * grad_U(q)
        trajectory[-1] = (q.copy(), p.copy())
        
        return q, p, trajectory
    
    # 2次元正規分布の負の勾配（力）
    def grad_U_gaussian(q, mu=np.array([0, 0]), cov_inv=np.eye(2)):
        return cov_inv @ (q - mu)
    
    # パラメータ
    mu = np.array([0, 0])
    cov = np.array([[1, 0.8], [0.8, 1]])
    cov_inv = np.linalg.inv(cov)
    grad_U = lambda q: grad_U_gaussian(q, mu, cov_inv)
    
    # 異なる初期条件でのリープフロッグ軌跡
    initial_positions = [
        np.array([-2, -2]),
        np.array([2, -1]), 
        np.array([-1, 2]),
        np.array([1.5, 1.5])
    ]
    
    colors = ['red', 'blue', 'green', 'purple']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 軌跡の可視化
    for i, (q_init, color) in enumerate(zip(initial_positions, colors)):
        p_init = np.random.normal(0, 1, 2)
        q_final, p_final, trajectory = leapfrog_integrator(
            q_init, p_init, epsilon=0.3, L=15, grad_U=grad_U
        )
        
        trajectory = np.array(trajectory)
        q_traj = trajectory[:, 0]
        
        axes[0, 0].plot(q_traj[:, 0], q_traj[:, 1], 'o-', color=color, 
                       markersize=4, linewidth=1.5, alpha=0.7,
                       label=f'軌跡 {i+1}')
        axes[0, 0].plot(q_init[0], q_init[1], 's', color=color, markersize=8)
        axes[0, 0].plot(q_final[0], q_final[1], '^', color=color, markersize=8)
    
    # 真の分布の等高線
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mu, cov)
    axes[0, 0].contour(X, Y, rv.pdf(pos), colors='black', alpha=0.5, linewidths=1)
    
    axes[0, 0].set_title('リープフロッグ軌跡\n（異なる初期条件）')
    axes[0, 0].set_xlabel('q1')
    axes[0, 0].set_ylabel('q2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # ランダムウォークとの比較
    def random_walk_steps(q_init, steps, step_size):
        """ランダムウォークのステップ"""
        q = q_init.copy()
        trajectory = [q.copy()]
        
        for _ in range(steps):
            step = np.random.normal(0, step_size, 2)
            q = q + step
            trajectory.append(q.copy())
        
        return np.array(trajectory)
    
    rw_traj = random_walk_steps(initial_positions[0], 15, 0.5)
    hmc_q_init = initial_positions[0]
    hmc_p_init = np.random.normal(0, 1, 2)
    _, _, hmc_trajectory = leapfrog_integrator(
        hmc_q_init, hmc_p_init, epsilon=0.3, L=15, grad_U=grad_U
    )
    hmc_traj = np.array([traj[0] for traj in hmc_trajectory])
    
    axes[0, 1].plot(rw_traj[:, 0], rw_traj[:, 1], 'ro-', 
                   markersize=4, linewidth=1.5, alpha=0.7, label='ランダムウォーク')
    axes[0, 1].plot(hmc_traj[:, 0], hmc_traj[:, 1], 'bo-', 
                   markersize=4, linewidth=1.5, alpha=0.7, label='HMC')
    axes[0, 1].contour(X, Y, rv.pdf(pos), colors='black', alpha=0.5, linewidths=1)
    axes[0, 1].set_title('ランダムウォーク vs HMC')
    axes[0, 1].set_xlabel('q1')
    axes[0, 1].set_ylabel('q2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # ステップサイズの効果
    epsilons = [0.1, 0.3, 0.5, 0.8]
    for i, eps in enumerate(epsilons):
        q_init = np.array([-2, -1])
        p_init = np.array([1, 0.5])
        _, _, traj = leapfrog_integrator(q_init, p_init, eps, 10, grad_U)
        traj = np.array([t[0] for t in traj])
        
        axes[1, 0].plot(traj[:, 0], traj[:, 1], 'o-', 
                       markersize=3, linewidth=1, alpha=0.8,
                       label=f'ε = {eps}')
    
    axes[1, 0].contour(X, Y, rv.pdf(pos), colors='black', alpha=0.5, linewidths=1)
    axes[1, 0].set_title('ステップサイズの効果')
    axes[1, 0].set_xlabel('q1')
    axes[1, 0].set_ylabel('q2')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    # エネルギー保存の確認
    q_init = np.array([1, 1])
    p_init = np.array([0.5, -0.5])
    
    def compute_hamiltonian(q, p, mu, cov_inv):
        U = 0.5 * (q - mu).T @ cov_inv @ (q - mu)
        K = 0.5 * p.T @ p
        return U + K
    
    _, _, traj = leapfrog_integrator(q_init, p_init, 0.2, 20, grad_U)
    
    hamiltonians = []
    for q, p in traj:
        H = compute_hamiltonian(q, p, mu, cov_inv)
        hamiltonians.append(H)
    
    axes[1, 1].plot(hamiltonians, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_title('ハミルトニアン（エネルギー）の保存')
    axes[1, 1].set_xlabel('リープフロッグステップ')
    axes[1, 1].set_ylabel('H(q,p)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("観察のポイント：")
    print("• HMCは勾配を利用して効率的に移動")
    print("• ランダムウォークより大きなステップが可能")
    print("• ステップサイズが性能に大きく影響")
    print("• ハミルトニアンの保存が重要")

hmc_intuition_step2()
```

### 3.3 Step 3: メトロポリス受理判定

```python
def hmc_intuition_step3():
    """Step 3: メトロポリス受理による誤差補正"""
    
    print("HMC Step 3: メトロポリス受理判定")
    print("=" * 50)
    print()
    print("1. リープフロッグ積分は近似 → 誤差が蓄積")
    print("2. メトロポリス受理で誤差を補正")
    print("3. ハミルトニアン差 ΔH を受理確率に使用")
    print("4. 理論的に正確なサンプリングを保証")
    print()
    
    def hamiltonian_monte_carlo_step(q_current, mu, cov_inv, epsilon, L):
        """1回のHMCステップ"""
        # 運動量のサンプリング
        p_current = np.random.normal(0, 1, len(q_current))
        
        # 現在のハミルトニアン
        def compute_H(q, p):
            U = 0.5 * (q - mu).T @ cov_inv @ (q - mu)
            K = 0.5 * p.T @ p
            return U + K
        
        H_current = compute_H(q_current, p_current)
        
        # リープフロッグ積分
        def grad_U(q):
            return cov_inv @ (q - mu)
        
        q, p = q_current.copy(), p_current.copy()
        
        # 最初の半ステップ
        p = p + 0.5 * epsilon * grad_U(q)
        
        for _ in range(L):
            # 位置の更新
            q = q + epsilon * p
            # 運動量の更新（最後以外）
            if _ < L - 1:
                p = p + epsilon * grad_U(q)
        
        # 最後の半ステップ
        p = p + 0.5 * epsilon * grad_U(q)
        
        # 運動量を反転（ハミルトニアン力学の時間反転対称性）
        p = -p
        
        # 提案後のハミルトニアン
        H_proposed = compute_H(q, p)
        
        # メトロポリス受理確率
        delta_H = H_proposed - H_current
        accept_prob = min(1.0, np.exp(-delta_H))
        
        # 受理/棄却
        if np.random.rand() < accept_prob:
            return q, True, delta_H, accept_prob
        else:
            return q_current, False, delta_H, accept_prob
    
    # パラメータ設定
    mu = np.array([0, 0])
    cov = np.array([[1, 0.8], [0.8, 1]])
    cov_inv = np.linalg.inv(cov)
    
    # 異なるパラメータでの受理率実験
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    L_values = [10, 20, 30]
    
    results = {}
    
    for L in L_values:
        accept_rates = []
        avg_delta_H = []
        
        for epsilon in epsilons:
            q = np.array([0.0, 0.0])
            accepted = 0
            delta_Hs = []
            
            for _ in range(200):
                q, accepted_step, delta_H, _ = hamiltonian_monte_carlo_step(
                    q, mu, cov_inv, epsilon, L
                )
                if accepted_step:
                    accepted += 1
                delta_Hs.append(abs(delta_H))
            
            accept_rates.append(accepted / 200)
            avg_delta_H.append(np.mean(delta_Hs))
        
        results[L] = {
            'accept_rates': accept_rates,
            'avg_delta_H': avg_delta_H
        }
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 受理率 vs ステップサイズ
    for L in L_values:
        axes[0, 0].plot(epsilons, results[L]['accept_rates'], 'o-', 
                       linewidth=2, markersize=6, label=f'L = {L}')
    
    axes[0, 0].axhline(0.8, color='red', linestyle='--', alpha=0.7, 
                      label='理想的受理率（~80%）')
    axes[0, 0].set_xlabel('ステップサイズ ε')
    axes[0, 0].set_ylabel('受理率')
    axes[0, 0].set_title('受理率 vs ステップサイズ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.05)
    
    # ハミルトニアン誤差 vs ステップサイズ
    for L in L_values:
        axes[0, 1].plot(epsilons, results[L]['avg_delta_H'], 'o-', 
                       linewidth=2, markersize=6, label=f'L = {L}')
    
    axes[0, 1].set_xlabel('ステップサイズ ε')
    axes[0, 1].set_ylabel('平均|ΔH|')
    axes[0, 1].set_title('ハミルトニアン誤差 vs ステップサイズ')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 実際のHMCサンプリング実行例
    q = np.array([0.0, 0.0])
    samples = []
    accept_history = []
    delta_H_history = []
    
    for i in range(1000):
        q, accepted, delta_H, accept_prob = hamiltonian_monte_carlo_step(
            q, mu, cov_inv, epsilon=0.25, L=20
        )
        samples.append(q.copy())
        accept_history.append(accepted)
        delta_H_history.append(delta_H)
    
    samples = np.array(samples)
    
    # サンプル結果
    axes[1, 0].scatter(samples[::5, 0], samples[::5, 1], alpha=0.6, s=10)
    
    # 真の分布の等高線
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mu, cov)
    axes[1, 0].contour(X, Y, rv.pdf(pos), colors='red', alpha=0.8, linewidths=2)
    
    axes[1, 0].set_title('HMCサンプリング結果')
    axes[1, 0].set_xlabel('q1')
    axes[1, 0].set_ylabel('q2')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 受理率の時系列
    window = 50
    rolling_accept = []
    for i in range(window, len(accept_history)):
        rolling_accept.append(np.mean(accept_history[i-window:i]))
    
    axes[1, 1].plot(rolling_accept, 'b-', linewidth=1.5)
    axes[1, 1].axhline(np.mean(accept_history), color='red', linestyle='--', 
                      label=f'平均受理率: {np.mean(accept_history):.2f}')
    axes[1, 1].set_xlabel('イテレーション')
    axes[1, 1].set_ylabel('受理率（移動平均）')
    axes[1, 1].set_title('受理率の時間変化')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"実験結果：")
    print(f"• 最終サンプル数: {len(samples)}")
    print(f"• 平均受理率: {np.mean(accept_history):.3f}")
    print(f"• 平均|ΔH|: {np.mean(np.abs(delta_H_history)):.6f}")
    print(f"• サンプル平均: [{np.mean(samples[:, 0]):.3f}, {np.mean(samples[:, 1]):.3f}]")
    print(f"• サンプル共分散:")
    print(np.cov(samples.T))

hmc_intuition_step3()
```

---

## 4. 実装から理解するHMC

### 4.1 完全なHMC実装

```python
class ComprehensiveHMC:
    """
    教育目的の包括的HMC実装
    デバッグ情報と詳細な説明付き
    """
    
    def __init__(self, log_prob_fn, grad_log_prob_fn, mass_matrix=None):
        """
        Parameters:
        - log_prob_fn: 対数確率密度関数 log π(q)
        - grad_log_prob_fn: 勾配関数 ∇log π(q)  
        - mass_matrix: 質量行列 M（デフォルトは単位行列）
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.mass_matrix = mass_matrix
        
        # 統計情報
        self.stats = {
            'n_accepted': 0,
            'n_proposed': 0,
            'hamiltonian_errors': [],
            'step_sizes_used': [],
            'accept_probs': []
        }
    
    def sample_momentum(self, dim):
        """運動量のサンプリング"""
        if self.mass_matrix is None:
            return np.random.normal(0, 1, dim)
        else:
            return np.random.multivariate_normal(
                np.zeros(dim), self.mass_matrix
            )
    
    def kinetic_energy(self, p):
        """運動エネルギーの計算"""
        if self.mass_matrix is None:
            return 0.5 * np.sum(p**2)
        else:
            return 0.5 * p.T @ np.linalg.solve(self.mass_matrix, p)
    
    def potential_energy(self, q):
        """ポテンシャルエネルギーの計算"""
        return -self.log_prob_fn(q)
    
    def hamiltonian(self, q, p):
        """ハミルトニアンの計算"""
        return self.potential_energy(q) + self.kinetic_energy(p)
    
    def leapfrog_step(self, q, p, epsilon):
        """リープフロッグ積分による1ステップ"""
        # 運動量の半ステップ更新
        p_half = p + 0.5 * epsilon * self.grad_log_prob_fn(q)
        
        # 位置の全ステップ更新
        if self.mass_matrix is None:
            q_new = q + epsilon * p_half
        else:
            q_new = q + epsilon * np.linalg.solve(self.mass_matrix, p_half)
        
        # 運動量の残り半ステップ更新
        p_new = p_half + 0.5 * epsilon * self.grad_log_prob_fn(q_new)
        
        return q_new, p_new
    
    def leapfrog_trajectory(self, q_initial, p_initial, epsilon, L):
        """完全なリープフロッグ軌跡の計算"""
        q, p = q_initial.copy(), p_initial.copy()
        trajectory = [(q.copy(), p.copy())]
        
        for step in range(L):
            q, p = self.leapfrog_step(q, p, epsilon)
            trajectory.append((q.copy(), p.copy()))
        
        return q, p, trajectory
    
    def hmc_step(self, q_current, epsilon, L, verbose=False):
        """
        1回のHMCステップ
        
        Returns:
        - q_new: 新しい位置
        - accepted: 受理されたかどうか
        - info: 詳細情報辞書
        """
        dim = len(q_current)
        
        # 1. 運動量のサンプリング
        p_current = self.sample_momentum(dim)
        
        # 2. 現在のハミルトニアン
        H_current = self.hamiltonian(q_current, p_current)
        
        # 3. リープフロッグ積分
        q_proposed, p_proposed, trajectory = self.leapfrog_trajectory(
            q_current, p_current, epsilon, L
        )
        
        # 4. 運動量の反転（時間反転対称性）
        p_proposed = -p_proposed
        
        # 5. 提案後のハミルトニアン
        H_proposed = self.hamiltonian(q_proposed, p_proposed)
        
        # 6. ハミルトニアン差
        delta_H = H_proposed - H_current
        
        # 7. メトロポリス受理確率
        accept_prob = min(1.0, np.exp(-delta_H))
        
        # 8. 受理/棄却
        self.stats['n_proposed'] += 1
        if np.random.rand() < accept_prob:
            q_new = q_proposed
            accepted = True
            self.stats['n_accepted'] += 1
        else:
            q_new = q_current
            accepted = False
        
        # 9. 統計情報の更新
        self.stats['hamiltonian_errors'].append(abs(delta_H))
        self.stats['step_sizes_used'].append(epsilon)
        self.stats['accept_probs'].append(accept_prob)
        
        # 10. 詳細情報
        info = {
            'H_current': H_current,
            'H_proposed': H_proposed,
            'delta_H': delta_H,
            'accept_prob': accept_prob,
            'accepted': accepted,
            'trajectory': trajectory,
            'momentum_initial': p_current,
            'momentum_final': p_proposed
        }
        
        if verbose:
            print(f"HMCステップ詳細:")
            print(f"  現在位置: {q_current}")
            print(f"  提案位置: {q_proposed}")
            print(f"  ΔH = {delta_H:.6f}")
            print(f"  受理確率 = {accept_prob:.6f}")
            print(f"  結果: {'受理' if accepted else '棄却'}")
            print()
        
        return q_new, accepted, info
    
    def sample(self, initial_q, n_samples, epsilon, L, verbose=False):
        """
        HMCサンプリングの実行
        
        Parameters:
        - initial_q: 初期位置
        - n_samples: サンプル数
        - epsilon: ステップサイズ
        - L: リープフロッグステップ数
        - verbose: 詳細出力フラグ
        
        Returns:
        - samples: サンプル配列
        - detailed_info: 各ステップの詳細情報
        """
        dim = len(initial_q)
        samples = np.zeros((n_samples, dim))
        detailed_info = []
        
        q_current = initial_q.copy()
        
        for i in range(n_samples):
            q_current, accepted, info = self.hmc_step(
                q_current, epsilon, L, verbose and i < 5  # 最初の5ステップのみ詳細出力
            )
            
            samples[i] = q_current
            detailed_info.append(info)
            
            if (i + 1) % 100 == 0:
                current_accept_rate = self.stats['n_accepted'] / self.stats['n_proposed']
                print(f"Iteration {i+1}/{n_samples}, "
                      f"Acceptance rate: {current_accept_rate:.3f}, "
                      f"Mean |ΔH|: {np.mean(self.stats['hamiltonian_errors'][-100:]):.6f}")
        
        return samples, detailed_info
    
    def get_statistics(self):
        """統計情報の取得"""
        if self.stats['n_proposed'] == 0:
            return {}
        
        return {
            'acceptance_rate': self.stats['n_accepted'] / self.stats['n_proposed'],
            'mean_hamiltonian_error': np.mean(self.stats['hamiltonian_errors']),
            'std_hamiltonian_error': np.std(self.stats['hamiltonian_errors']),
            'mean_accept_prob': np.mean(self.stats['accept_probs']),
            'n_samples': self.stats['n_proposed']
        }

# HMC実装のテスト
def test_comprehensive_hmc():
    """包括的HMC実装のテスト"""
    
    # 目標分布：2次元正規分布
    mu = np.array([1.0, -0.5])
    cov = np.array([[2.0, 1.2], [1.2, 1.5]])
    cov_inv = np.linalg.inv(cov)
    
    def log_prob(q):
        diff = q - mu
        return -0.5 * diff.T @ cov_inv @ diff
    
    def grad_log_prob(q):
        return -cov_inv @ (q - mu)
    
    # HMCサンプラーの作成
    hmc = ComprehensiveHMC(log_prob, grad_log_prob)
    
    print("=== HMC実装テスト開始 ===")
    print()
    
    # サンプリング実行
    initial_q = np.array([0.0, 0.0])
    samples, detailed_info = hmc.sample(
        initial_q=initial_q,
        n_samples=1000,
        epsilon=0.2,
        L=25,
        verbose=True  # 最初の5ステップの詳細を表示
    )
    
    # 統計情報の表示
    stats = hmc.get_statistics()
    print("\n=== 最終統計 ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # 結果の可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # サンプル散布図
    burnin = 100
    clean_samples = samples[burnin:]
    
    axes[0, 0].scatter(clean_samples[:, 0], clean_samples[:, 1], alpha=0.6, s=10)
    axes[0, 0].scatter(mu[0], mu[1], color='red', s=100, marker='x', 
                      linewidth=3, label='真の平均')
    axes[0, 0].set_title('HMCサンプル')
    axes[0, 0].set_xlabel('q1')
    axes[0, 0].set_ylabel('q2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # トレースプロット
    axes[0, 1].plot(samples[:500, 0], alpha=0.8, label='q1', linewidth=0.8)
    axes[0, 1].plot(samples[:500, 1], alpha=0.8, label='q2', linewidth=0.8)
    axes[0, 1].axvline(burnin, color='red', linestyle='--', alpha=0.7, label='Burn-in')
    axes[0, 1].set_title('トレースプロット')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ハミルトニアン誤差
    errors = [info['delta_H'] for info in detailed_info]
    axes[0, 2].plot(np.abs(errors[:500]), alpha=0.8, linewidth=0.8)
    axes[0, 2].set_title('ハミルトニアン誤差 |ΔH|')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('|ΔH|')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 受理確率の分布
    accept_probs = [info['accept_prob'] for info in detailed_info]
    axes[1, 0].hist(accept_probs, bins=30, alpha=0.7, density=True)
    axes[1, 0].set_title('受理確率の分布')
    axes[1, 0].set_xlabel('受理確率')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 軌跡の例
    sample_trajectory = detailed_info[50]['trajectory']  # 50番目のステップの軌跡
    traj_q = np.array([traj[0] for traj in sample_trajectory])
    
    axes[1, 1].plot(traj_q[:, 0], traj_q[:, 1], 'bo-', markersize=4, 
                   linewidth=1.5, alpha=0.7)
    axes[1, 1].plot(traj_q[0, 0], traj_q[0, 1], 'go', markersize=8, label='開始')
    axes[1, 1].plot(traj_q[-1, 0], traj_q[-1, 1], 'ro', markersize=8, label='終了')
    axes[1, 1].set_title('リープフロッグ軌跡の例')
    axes[1, 1].set_xlabel('q1')
    axes[1, 1].set_ylabel('q2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    # 統計比較
    sample_mean = np.mean(clean_samples, axis=0)
    sample_cov = np.cov(clean_samples.T)
    
    comparison_data = [
        ['平均 q1', mu[0], sample_mean[0], abs(mu[0] - sample_mean[0])],
        ['平均 q2', mu[1], sample_mean[1], abs(mu[1] - sample_mean[1])],
        ['分散 q1', cov[0,0], sample_cov[0,0], abs(cov[0,0] - sample_cov[0,0])],
        ['分散 q2', cov[1,1], sample_cov[1,1], abs(cov[1,1] - sample_cov[1,1])],
        ['共分散', cov[0,1], sample_cov[0,1], abs(cov[0,1] - sample_cov[0,1])]
    ]
    
    axes[1, 2].axis('off')
    table_text = "統計比較\n\n"
    table_text += f"{'統計量':<8} {'真値':<8} {'推定値':<8} {'誤差':<8}\n"
    table_text += "-" * 40 + "\n"
    for row in comparison_data:
        table_text += f"{row[0]:<8} {row[1]:<8.3f} {row[2]:<8.3f} {row[3]:<8.3f}\n"
    
    axes[1, 2].text(0.1, 0.8, table_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return samples, detailed_info, hmc

# テスト実行
samples, detailed_info, hmc_instance = test_comprehensive_hmc()
```

---

## 5. HMC vs 従来手法：性能比較の詳細分析

### 5.1 多次元での性能比較

```python
def multidimensional_comparison():
    """多次元での性能比較"""
    
    def create_test_distribution(dim, correlation=0.8):
        """テスト用多次元正規分布の作成"""
        # 強い相関を持つ共分散行列
        cov = np.eye(dim)
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    cov[i, j] = correlation * np.exp(-0.1 * abs(i - j))
        
        mu = np.zeros(dim)
        cov_inv = np.linalg.inv(cov)
        
        def log_prob(q):
            diff = q - mu
            return -0.5 * diff.T @ cov_inv @ diff
        
        def grad_log_prob(q):
            return -cov_inv @ (q - mu)
        
        return mu, cov, log_prob, grad_log_prob
    
    def random_walk_mh(log_prob_fn, initial_q, n_samples, step_size):
        """ランダムウォークMH法"""
        dim = len(initial_q)
        samples = np.zeros((n_samples, dim))
        current = initial_q.copy()
        current_log_prob = log_prob_fn(current)
        n_accepted = 0
        
        for i in range(n_samples):
            # 提案
            proposed = current + np.random.normal(0, step_size, dim)
            proposed_log_prob = log_prob_fn(proposed)
            
            # 受理確率
            log_alpha = proposed_log_prob - current_log_prob
            alpha = min(1.0, np.exp(log_alpha))
            
            # 受理/棄却
            if np.random.rand() < alpha:
                current = proposed
                current_log_prob = proposed_log_prob
                n_accepted += 1
            
            samples[i] = current
        
        return samples, n_accepted / n_samples
    
    def compute_autocorr_time(samples, max_lag=None):
        """自己相関時間の計算"""
        if max_lag is None:
            max_lag = min(len(samples) // 4, 200)
        
        autocorr_times = []
        for dim in range(samples.shape[1]):
            data = samples[:, dim]
            data = data - np.mean(data)
            
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # 積分自己相関時間
            tau_int = 1.0
            for lag in range(1, min(len(autocorr), max_lag)):
                if autocorr[lag] > 0.01:
                    tau_int += 2 * autocorr[lag]
                else:
                    break
            
            autocorr_times.append(tau_int)
        
        return np.mean(autocorr_times)
    
    # 次元数を変化させて比較
    dimensions = [2, 5, 10, 20, 50]
    methods = ['Random Walk MH', 'HMC']
    
    results = {dim: {} for dim in dimensions}
    
    for dim in dimensions:
        print(f"\n=== {dim}次元での比較 ===")
        
        # テスト分布の作成
        mu, cov, log_prob, grad_log_prob = create_test_distribution(dim)
        
        # 共通パラメータ
        n_samples = 2000
        initial_q = np.zeros(dim)
        
        # Random Walk MH
        print("Random Walk MH実行中...")
        step_size = 0.8 / np.sqrt(dim)  # 次元に応じてスケーリング
        rwmh_samples, rwmh_accept_rate = random_walk_mh(
            log_prob, initial_q, n_samples, step_size
        )
        rwmh_autocorr_time = compute_autocorr_time(rwmh_samples[200:])
        
        # HMC
        print("HMC実行中...")
        hmc = ComprehensiveHMC(log_prob, grad_log_prob)
        epsilon = 0.1 / np.sqrt(dim)  # 次元に応じてスケーリング
        L = max(10, int(20 / np.sqrt(dim)))  # 次元に応じて調整
        
        hmc_samples, _ = hmc.sample(initial_q, n_samples, epsilon, L)
        hmc_stats = hmc.get_statistics()
        hmc_autocorr_time = compute_autocorr_time(hmc_samples[200:])
        
        # 結果の保存
        results[dim]['RWMH'] = {
            'accept_rate': rwmh_accept_rate,
            'autocorr_time': rwmh_autocorr_time,
            'eff_samples': len(rwmh_samples[200:]) / (2 * rwmh_autocorr_time + 1)
        }
        
        results[dim]['HMC'] = {
            'accept_rate': hmc_stats['acceptance_rate'],
            'autocorr_time': hmc_autocorr_time,
            'eff_samples': len(hmc_samples[200:]) / (2 * hmc_autocorr_time + 1)
        }
        
        print(f"RWMH: 受理率={rwmh_accept_rate:.3f}, τ={rwmh_autocorr_time:.2f}")
        print(f"HMC:  受理率={hmc_stats['acceptance_rate']:.3f}, τ={hmc_autocorr_time:.2f}")
    
    # 結果の可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 受理率
    rwmh_accept_rates = [results[dim]['RWMH']['accept_rate'] for dim in dimensions]
    hmc_accept_rates = [results[dim]['HMC']['accept_rate'] for dim in dimensions]
    
    axes[0, 0].plot(dimensions, rwmh_accept_rates, 'ro-', linewidth=2, 
                   markersize=6, label='Random Walk MH')
    axes[0, 0].plot(dimensions, hmc_accept_rates, 'bo-', linewidth=2, 
                   markersize=6, label='HMC')
    axes[0, 0].set_xlabel('次元数')
    axes[0, 0].set_ylabel('受理率')
    axes[0, 0].set_title('受理率 vs 次元数')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 自己相関時間
    rwmh_autocorr_times = [results[dim]['RWMH']['autocorr_time'] for dim in dimensions]
    hmc_autocorr_times = [results[dim]['HMC']['autocorr_time'] for dim in dimensions]
    
    axes[0, 1].plot(dimensions, rwmh_autocorr_times, 'ro-', linewidth=2, 
                   markersize=6, label='Random Walk MH')
    axes[0, 1].plot(dimensions, hmc_autocorr_times, 'bo-', linewidth=2, 
                   markersize=6, label='HMC')
    axes[0, 1].set_xlabel('次元数')
    axes[0, 1].set_ylabel('自己相関時間')
    axes[0, 1].set_title('自己相関時間 vs 次元数')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 有効サンプルサイズ
    rwmh_eff_samples = [results[dim]['RWMH']['eff_samples'] for dim in dimensions]
    hmc_eff_samples = [results[dim]['HMC']['eff_samples'] for dim in dimensions]
    
    axes[1, 0].plot(dimensions, rwmh_eff_samples, 'ro-', linewidth=2, 
                   markersize=6, label='Random Walk MH')
    axes[1, 0].plot(dimensions, hmc_eff_samples, 'bo-', linewidth=2, 
                   markersize=6, label='HMC')
    axes[1, 0].set_xlabel('次元数')
    axes[1, 0].set_ylabel('有効サンプルサイズ')
    axes[1, 0].set_title('有効サンプルサイズ vs 次元数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 効率比（HMC / RWMH）
    efficiency_ratios = [
        results[dim]['HMC']['eff_samples'] / results[dim]['RWMH']['eff_samples']
        for dim in dimensions
    ]
    
    axes[1, 1].plot(dimensions, efficiency_ratios, 'go-', linewidth=2, 
                   markersize=6, label='HMC効率 / RWMH効率')
    axes[1, 1].axhline(1, color='black', linestyle='--', alpha=0.7, label='等価')
    axes[1, 1].set_xlabel('次元数')
    axes[1, 1].set_ylabel('効率比')
    axes[1, 1].set_title('HMCの相対効率')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 数値結果のサマリー
    print("\n=== 性能比較サマリー ===")
    print(f"{'次元':<4} {'方法':<12} {'受理率':<8} {'τ':<8} {'ESS':<8} {'効率比':<8}")
    print("-" * 60)
    
    for dim in dimensions:
        for method in ['RWMH', 'HMC']:
            data = results[dim][method]
            if method == 'HMC':
                ratio = results[dim]['HMC']['eff_samples'] / results[dim]['RWMH']['eff_samples']
                print(f"{dim:<4} {method:<12} {data['accept_rate']:<8.3f} "
                      f"{data['autocorr_time']:<8.2f} {data['eff_samples']:<8.1f} {ratio:<8.2f}")
            else:
                print(f"{dim:<4} {method:<12} {data['accept_rate']:<8.3f} "
                      f"{data['autocorr_time']:<8.2f} {data['eff_samples']:<8.1f} {'1.00':<8}")
    
    return results

# 多次元比較の実行
multidim_results = multidimensional_comparison()
```

---

## 6. 実践的パラメータチューニング

### 6.1 ステップサイズとリープフロッグステップ数の最適化

```python
def hmc_parameter_optimization():
    """HMCパラメータの体系的最適化"""
    
    # 目標分布の設定
    dim = 5
    mu = np.zeros(dim)
    # 病的な条件数を持つ共分散行列
    eigenvalues = np.array([10.0, 5.0, 1.0, 0.5, 0.1])
    eigenvectors = np.random.randn(dim, dim)
    eigenvectors, _ = np.linalg.qr(eigenvectors)
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cov_inv = np.linalg.inv(cov)
    
    def log_prob(q):
        diff = q - mu
        return -0.5 * diff.T @ cov_inv @ diff
    
    def grad_log_prob(q):
        return -cov_inv @ (q - mu)
    
    print("目標分布の条件数:", np.linalg.cond(cov))
    print("固有値:", eigenvalues)
    print()
    
    # パラメータグリッドの設定
    epsilons = np.logspace(-2, 0, 8)  # 0.01 から 1.0
    L_values = [5, 10, 20, 30, 50]
    
    # 結果保存用
    results_grid = np.zeros((len(epsilons), len(L_values), 4))  # accept_rate, autocorr_time, ess, mean_error
    
    print("パラメータ最適化実行中...")
    print(f"グリッド サイズ: {len(epsilons)} × {len(L_values)} = {len(epsilons) * len(L_values)} 組み合わせ")
    print()
    
    total_combinations = len(epsilons) * len(L_values)
    combination_count = 0
    
    for i, epsilon in enumerate(epsilons):
        for j, L in enumerate(L_values):
            combination_count += 1
            
            # HMCサンプリング
            hmc = ComprehensiveHMC(log_prob, grad_log_prob)
            samples, _ = hmc.sample(
                initial_q=np.zeros(dim),
                n_samples=1000,
                epsilon=epsilon,
                L=L
            )
            
            stats = hmc.get_statistics()
            
            # 自己相関時間の計算
            burnin = 200
            clean_samples = samples[burnin:]
            
            autocorr_times = []
            for d in range(dim):
                data = clean_samples[:, d]
                data = data - np.mean(data)
                autocorr = np.correlate(data, data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                tau_int = 1.0
                for lag in range(1, min(len(autocorr), 100)):
                    if autocorr[lag] > 0.01:
                        tau_int += 2 * autocorr[lag]
                    else:
                        break
                autocorr_times.append(tau_int)
            
            mean_autocorr_time = np.mean(autocorr_times)
            ess = len(clean_samples) / (2 * mean_autocorr_time + 1)
            
            # 結果の保存
            results_grid[i, j, 0] = stats['acceptance_rate']
            results_grid[i, j, 1] = mean_autocorr_time
            results_grid[i, j, 2] = ess
            results_grid[i, j, 3] = stats['mean_hamiltonian_error']
            
            if combination_count % 5 == 0:
                print(f"進捗: {combination_count}/{total_combinations} "
                      f"({100*combination_count/total_combinations:.1f}%)")
    
    # 結果の可視化
    fig = plt.figure(figsize=(16, 12))
    
    # ヒートマップ用のデータ準備
    epsilon_labels = [f"{eps:.3f}" for eps in epsilons]
    L_labels = [str(L) for L in L_values]
    
    # 1. 受理率
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(results_grid[:, :, 0], aspect='auto', cmap='viridis', 
                     vmin=0, vmax=1)
    ax1.set_title('受理率')
    ax1.set_xlabel('リープフロッグステップ数 L')
    ax1.set_ylabel('ステップサイズ ε')
    ax1.set_xticks(range(len(L_values)))
    ax1.set_xticklabels(L_labels)
    ax1.set_yticks(range(0, len(epsilons), 2))
    ax1.set_yticklabels([epsilon_labels[i] for i in range(0, len(epsilons), 2)])
    plt.colorbar(im1, ax=ax1)
    
    # 2. 自己相関時間
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(np.log10(results_grid[:, :, 1]), aspect='auto', cmap='viridis_r')
    ax2.set_title('自己相関時間 (log10)')
    ax2.set_xlabel('リープフロッグステップ数 L')
    ax2.set_ylabel('ステップサイズ ε')
    ax2.set_xticks(range(len(L_values)))
    ax2.set_xticklabels(L_labels)
    ax2.set_yticks(range(0, len(epsilons), 2))
    ax2.set_yticklabels([epsilon_labels[i] for i in range(0, len(epsilons), 2)])
    plt.colorbar(im2, ax=ax2)
    
    # 3. 有効サンプルサイズ
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(results_grid[:, :, 2], aspect='auto', cmap='viridis')
    ax3.set_title('有効サンプルサイズ')
    ax3.set_xlabel('リープフロッグステップ数 L')
    ax3.set_ylabel('ステップサイズ ε')
    ax3.set_xticks(range(len(L_values)))
    ax3.set_xticklabels(L_labels)
    ax3.set_yticks(range(0, len(epsilons), 2))
    ax3.set_yticklabels([epsilon_labels[i] for i in range(0, len(epsilons), 2)])
    plt.colorbar(im3, ax=ax3)
    
    # 4. ハミルトニアン誤差
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(np.log10(results_grid[:, :, 3]), aspect='auto', cmap='viridis_r')
    ax4.set_title('ハミルトニアン誤差 (log10)')
    ax4.set_xlabel('リープフロッグステップ数 L')
    ax4.set_ylabel('ステップサイズ ε')
    ax4.set_xticks(range(len(L_values)))
    ax4.set_xticklabels(L_labels)
    ax4.set_yticks(range(0, len(epsilons), 2))
    ax4.set_yticklabels([epsilon_labels[i] for i in range(0, len(epsilons), 2)])
    plt.colorbar(im4, ax=ax4)
    
    # 5. 最適解の特定
    # 有効サンプルサイズを最大化する組み合わせ
    best_i, best_j = np.unravel_index(np.argmax(results_grid[:, :, 2]), 
                                      results_grid[:, :, 2].shape)
    best_epsilon = epsilons[best_i]
    best_L = L_values[best_j]
    
    # パレート最適解の可視化
    ax5 = plt.subplot(2, 3, 5)
    
    # 全組み合わせをプロット
    accept_rates_flat = results_grid[:, :, 0].flatten()
    ess_flat = results_grid[:, :, 2].flatten()
    
    scatter = ax5.scatter(accept_rates_flat, ess_flat, c=results_grid[:, :, 1].flatten(), 
                         cmap='viridis_r', alpha=0.7, s=30)
    ax5.scatter(results_grid[best_i, best_j, 0], results_grid[best_i, best_j, 2], 
               color='red', s=100, marker='*', label=f'最適解 (ε={best_epsilon:.3f}, L={best_L})')
    ax5.set_xlabel('受理率')
    ax5.set_ylabel('有効サンプルサイズ')
    ax5.set_title('パフォーマンス散布図')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='自己相関時間')
    
    # 6. 最適化サマリー
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 上位5つの組み合わせ
    ess_indices = np.unravel_index(np.argsort(results_grid[:, :, 2].flatten())[-5:],
                                   results_grid[:, :, 2].shape)
    
    summary_text = "最適パラメータ (ESS上位5位)\n\n"
    summary_text += f"{'順位':<4} {'ε':<8} {'L':<4} {'受理率':<8} {'τ':<8} {'ESS':<8}\n"
    summary_text += "-" * 50 + "\n"
    
    for rank, (i, j) in enumerate(zip(ess_indices[0][::-1], ess_indices[1][::-1])):
        eps = epsilons[i]
        L = L_values[j]
        accept_rate = results_grid[i, j, 0]
        autocorr_time = results_grid[i, j, 1]
        ess = results_grid[i, j, 2]
        
        summary_text += f"{rank+1:<4} {eps:<8.3f} {L:<4} {accept_rate:<8.3f} "
        summary_text += f"{autocorr_time:<8.2f} {ess:<8.1f}\n"
    
    summary_text += f"\n推奨設定:\n"
    summary_text += f"ε = {best_epsilon:.3f}\n"
    summary_text += f"L = {best_L}\n"
    summary_text += f"期待性能:\n"
    summary_text += f"  受理率: {results_grid[best_i, best_j, 0]:.3f}\n"
    summary_text += f"  ESS: {results_grid[best_i, best_j, 2]:.1f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    # 最適パラメータでの詳細テスト
    print(f"\n=== 最適パラメータでの詳細テスト ===")
    print(f"最適設定: ε = {best_epsilon:.3f}, L = {best_L}")
    
    hmc_optimal = ComprehensiveHMC(log_prob, grad_log_prob)
    optimal_samples, optimal_info = hmc_optimal.sample(
        initial_q=np.zeros(dim),
        n_samples=3000,
        epsilon=best_epsilon,
        L=best_L
    )
    
    optimal_stats = hmc_optimal.get_statistics()
    print("最適パラメータ性能:")
    for key, value in optimal_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    return results_grid, epsilons, L_values, best_epsilon, best_L

# パラメータ最適化の実行
optimization_results = hmc_parameter_optimization()
```

---

## 7. 高度な応用と拡張

### 7.1 階層ベイズモデルでのHMC応用

```python
def hierarchical_bayes_hmc_example():
    """階層ベイズモデルでのHMC応用例"""
    
    print("=== 階層ベイズモデル例：学校の成績分析 ===")
    print()
    
    # データ生成（８つの学校の成績データ）
    np.random.seed(42)
    
    # 真のパラメータ
    n_schools = 8
    true_mu = 5.0  # 全体平均
    true_tau = 2.0  # 学校間のばらつき
    true_theta = np.random.normal(true_mu, true_tau, n_schools)  # 各学校の真の平均
    school_sigmas = np.array([2.0, 1.5, 2.5, 1.8, 2.2, 1.6, 2.0, 1.9])  # 既知の学校内標準偏差
    
    # 観測データ
    school_sizes = [20, 15, 25, 18, 22, 16, 20, 19]
    observed_data = []
    
    for i in range(n_schools):
        school_data = np.random.normal(true_theta[i], school_sigmas[i], school_sizes[i])
        observed_data.append(school_data)
    
    # 観測統計量
    school_means = [np.mean(data) for data in observed_data]
    
    print("観測データサマリー:")
    print(f"{'学校':<4} {'サイズ':<6} {'観測平均':<10} {'真の平均':<10} {'標準偏差':<10}")
    print("-" * 50)
    for i in range(n_schools):
        print(f"{i+1:<4} {school_sizes[i]:<6} {school_means[i]:<10.3f} {true_theta[i]:<10.3f} {school_sigmas[i]:<10.3f}")
    print()
    
    # 階層ベイズモデルの定義
    def hierarchical_log_prob(params):
        """
        階層ベイズモデルの対数事後確率
        params = [mu, log_tau, theta_1, ..., theta_n]
        """
        mu = params[0]
        log_tau = params[1] 
        tau = np.exp(log_tau)  # 正の制約のためlog変換
        theta = params[2:2+n_schools]
        
        log_prob = 0.0
        
        # 事前分布
        log_prob += -0.5 * (mu - 0)**2 / 100**2  # μ ~ N(0, 100)
        log_prob += -0.5 * (log_tau - 0)**2 / 10**2  # log(τ) ~ N(0, 10)
        
        # 階層構造: θ_i | μ, τ ~ N(μ, τ)
        for i in range(n_schools):
            log_prob += -0.5 * (theta[i] - mu)**2 / tau**2
        
        # 尤度: y_ij | θ_i ~ N(θ_i, σ_i)
        for i in range(n_schools):
            n_i = school_sizes[i]
            y_bar_i = school_means[i]
            sigma_i = school_sigmas[i]
            
            # 十分統計量を使った尤度
            log_prob += -0.5 * n_i * (y_bar_i - theta[i])**2 / sigma_i**2
        
        return log_prob
    
    def hierarchical_grad_log_prob(params):
        """階層ベイズモデルの勾配"""
        mu = params[0]
        log_tau = params[1]
        tau = np.exp(log_tau)
        theta = params[2:2+n_schools]
        
        grad = np.zeros_like(params)
        
        # ∂/∂μ
        grad[0] = -mu / 100**2  # 事前分布
        for i in range(n_schools):
            grad[0] += (theta[i] - mu) / tau**2  # 階層構造
        
        # ∂/∂log_τ
        grad[1] = -log_tau / 10**2  # 事前分布
        grad[1] += n_schools  # |dτ/d(log τ)| = τ からのヤコビアン
        for i in range(n_schools):
            grad[1] += -(theta[i] - mu)**2 / tau**2  # 階層構造
        
        # ∂/∂θ_i
        for i in range(n_schools):
            grad[2+i] = -(theta[i] - mu) / tau**2  # 階層構造
            
            n_i = school_sizes[i]
            y_bar_i = school_means[i]
            sigma_i = school_sigmas[i]
            grad[2+i] += n_i * (y_bar_i - theta[i]) / sigma_i**2  # 尤度
        
        return grad
    
    # HMCサンプリング
    print("階層ベイズモデルのHMCサンプリング実行中...")
    
    # 初期値
    initial_params = np.concatenate([
        [0.0],  # mu
        [0.0],  # log_tau
        school_means  # theta（観測平均で初期化）
    ])
    
    hmc = ComprehensiveHMC(hierarchical_log_prob, hierarchical_grad_log_prob)
    samples, detailed_info = hmc.sample(
        initial_q=initial_params,
        n_samples=3000,
        epsilon=0.02,
        L=30
    )
    
    stats = hmc.get_statistics()
    print(f"HMC統計: 受理率={stats['acceptance_rate']:.3f}")
    print()
    
    # 結果の分析
    burnin = 500
    clean_samples = samples[burnin:]
    
    # パラメータの抽出
    mu_samples = clean_samples[:, 0]
    log_tau_samples = clean_samples[:, 1]
    tau_samples = np.exp(log_tau_samples)
    theta_samples = clean_samples[:, 2:2+n_schools]
    
    # 事後統計
    print("事後統計サマリー:")
    print(f"{'パラメータ':<12} {'真値':<10} {'事後平均':<12} {'95%信頼区間':<20}")
    print("-" * 60)
    
    # μ
    mu_mean = np.mean(mu_samples)
    mu_ci = np.percentile(mu_samples, [2.5, 97.5])
    print(f"{'μ (全体平均)':<12} {true_mu:<10.3f} {mu_mean:<12.3f} [{mu_ci[0]:.3f}, {mu_ci[1]:.3f}]")
    
    # τ
    tau_mean = np.mean(tau_samples)
    tau_ci = np.percentile(tau_samples, [2.5, 97.5])
    print(f"{'τ (学校間SD)':<12} {true_tau:<10.3f} {tau_mean:<12.3f} [{tau_ci[0]:.3f}, {tau_ci[1]:.3f}]")
    
    # θ
    for i in range(n_schools):
        theta_mean = np.mean(theta_samples[:, i])
        theta_ci = np.percentile(theta_samples[:, i], [2.5, 97.5])
        print(f"{'θ_' + str(i+1):<12} {true_theta[i]:<10.3f} {theta_mean:<12.3f} [{theta_ci[0]:.3f}, {theta_ci[1]:.3f}]")
    
    # 可視化
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # μの事後分布
    axes[0, 0].hist(mu_samples, bins=50, density=True, alpha=0.7, color='blue')
    axes[0, 0].axvline(true_mu, color='red', linestyle='--', linewidth=2, label='真値')
    axes[0, 0].axvline(mu_mean, color='green', linestyle='-', linewidth=2, label='事後平均')
    axes[0, 0].set_title('μ (全体平均) の事後分布')
    axes[0, 0].set_xlabel('μ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # τの事後分布
    axes[0, 1].hist(tau_samples, bins=50, density=True, alpha=0.7, color='blue')
    axes[0, 1].axvline(true_tau, color='red', linestyle='--', linewidth=2, label='真値')
    axes[0, 1].axvline(tau_mean, color='green', linestyle='-', linewidth=2, label='事後平均')
    axes[0, 1].set_title('τ (学校間標準偏差) の事後分布')
    axes[0, 1].set_xlabel('τ')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # トレースプロット
    axes[0, 2].plot(mu_samples[:1000], alpha=0.8, label='μ', linewidth=0.8)
    axes[0, 2].plot(tau_samples[:1000], alpha=0.8, label='τ', linewidth=0.8)
    axes[0, 2].set_title('μ, τ のトレースプロット')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 各学校のθの事後分布（最初の6校）
    for i in range(6):
        row, col = (i // 3) + 1, i % 3
        axes[row, col].hist(theta_samples[:, i], bins=30, density=True, 
                           alpha=0.7, color='blue')
        axes[row, col].axvline(true_theta[i], color='red', linestyle='--', 
                              linewidth=2, label='真値')
        axes[row, col].axvline(school_means[i], color='orange', linestyle=':', 
                              linewidth=2, label='観測平均')
        axes[row, col].axvline(np.mean(theta_samples[:, i]), color='green', 
                              linestyle='-', linewidth=2, label='事後平均')
        axes[row, col].set_title(f'学校{i+1}: θ_{i+1} の事後分布')
        axes[row, col].set_xlabel(f'θ_{i+1}')
        axes[row, col].set_ylabel('密度')
        if i == 0:
            axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # shrinkage効果の分析
    print("\n=== Shrinkage効果の分析 ===")
    print(f"{'学校':<4} {'観測平均':<10} {'事後平均':<10} {'Shrinkage':<12} {'真値との距離改善':<20}")
    print("-" * 65)
    
    for i in range(n_schools):
        obs_mean = school_means[i]
        post_mean = np.mean(theta_samples[:, i])
        shrinkage = abs(post_mean - obs_mean)
        
        # 真値との距離の比較
        obs_error = abs(obs_mean - true_theta[i])
        post_error = abs(post_mean - true_theta[i])
        improvement = obs_error - post_error
        
        print(f"{i+1:<4} {obs_mean:<10.3f} {post_mean:<10.3f} {shrinkage:<12.3f} {improvement:<20.3f}")
    
    # 予測分布の計算
    print("\n=== 新しい学校の予測 ===")
    
    # 新しい学校の予測分布: N(μ, τ)
    pred_samples = np.random.normal(mu_samples, tau_samples)
    pred_mean = np.mean(pred_samples)
    pred_ci = np.percentile(pred_samples, [2.5, 97.5])
    
    print(f"新しい学校の成績予測:")
    print(f"  予測平均: {pred_mean:.3f}")
    print(f"  95%予測区間: [{pred_ci[0]:.3f}, {pred_ci[1]:.3f}]")
    
    return samples, (mu_samples, tau_samples, theta_samples)

# 階層ベイズモデルの実行
hierarchical_samples, hierarchical_results = hierarchical_bayes_hmc_example()
```

---

## 8. よくある問題とデバッグ手法

### 8.1 HMCの典型的な問題と解決策

```python
def hmc_debugging_guide():
    """HMCの問題診断と解決ガイド"""
    
    print("=== HMC診断とデバッグガイド ===")
    print()
    
    def create_problematic_distributions():
        """問題のある分布の例を作成"""
        
        # 1. 高い条件数（病的な相関）
        def highly_correlated_dist():
            cov = np.array([[1.0, 0.99], [0.99, 1.0]])
            mu = np.array([0.0, 0.0])
            cov_inv = np.linalg.inv(cov)
            
            def log_prob(q):
                diff = q - mu
                return -0.5 * diff.T @ cov_inv @ diff
            
            def grad_log_prob(q):
                return -cov_inv @ (q - mu)
            
            return log_prob, grad_log_prob, "高条件数分布 (相関=0.99)"
        
        # 2. 重い裾を持つ分布
        def heavy_tailed_dist():
            def log_prob(q):
                # t分布 (自由度=3)
                return -2.0 * np.log(1 + np.sum(q**2) / 3)
            
            def grad_log_prob(q):
                return -4.0 * q / (3 + np.sum(q**2))
            
            return log_prob, grad_log_prob, "重い裾分布 (t分布)"
        
        # 3. マルチモーダル分布
        def multimodal_dist():
            def log_prob(q):
                # 2つのガウシアンの混合
                comp1 = -0.5 * np.sum((q - np.array([-2, 0]))**2)
                comp2 = -0.5 * np.sum((q - np.array([2, 0]))**2)
                return np.log(0.5 * np.exp(comp1) + 0.5 * np.exp(comp2))
            
            def grad_log_prob(q):
                # 数値微分で近似
                eps = 1e-6
                grad = np.zeros_like(q)
                for i in range(len(q)):
                    q_plus = q.copy()
                    q_minus = q.copy()
                    q_plus[i] += eps
                    q_minus[i] -= eps
                    grad[i] = (log_prob(q_plus) - log_prob(q_minus)) / (2 * eps)
                return grad
            
            return log_prob, grad_log_prob, "マルチモーダル分布"
        
        # 4. スケールの異なる次元
        def different_scales_dist():
            scales = np.array([10.0, 0.1])  # 非常に異なるスケール
            
            def log_prob(q):
                return -0.5 * np.sum((q / scales)**2)
            
            def grad_log_prob(q):
                return -q / (scales**2)
            
            return log_prob, grad_log_prob, "異なるスケール分布"
        
        return [
            highly_correlated_dist(),
            heavy_tailed_dist(),
            multimodal_dist(),
            different_scales_dist()
        ]
    
    def diagnose_hmc_problems(log_prob, grad_log_prob, distribution_name):
        """HMCの問題を診断"""
        
        print(f"\n--- {distribution_name} の診断 ---")
        
        # 複数のパラメータ設定でテスト
        test_configs = [
            {"epsilon": 0.01, "L": 10, "name": "小ステップ・短軌跡"},
            {"epsilon": 0.1, "L": 20, "name": "標準設定"},
            {"epsilon": 0.5, "L": 10, "name": "大ステップ・短軌跡"},
            {"epsilon": 0.1, "L": 50, "name": "標準ステップ・長軌跡"},
        ]
        
        results = {}
        
        for config in test_configs:
            hmc = ComprehensiveHMC(log_prob, grad_log_prob)
            
            try:
                samples, _ = hmc.sample(
                    initial_q=np.array([0.0, 0.0]),
                    n_samples=500,
                    epsilon=config["epsilon"],
                    L=config["L"]
                )
                
                stats = hmc.get_statistics()
                
                # 追加診断指標
                if len(samples) > 100:
                    # 有効サンプルサイズの計算
                    ess_estimates = []
                    for dim in range(samples.shape[1]):
                        data = samples[50:, dim]  # バーンイン除去
                        if len(data) > 50:
                            autocorr = np.correlate(data - np.mean(data), 
                                                   data - np.mean(data), mode='full')
                            autocorr = autocorr[len(autocorr)//2:]
                            autocorr = autocorr / autocorr[0]
                            
                            tau_int = 1.0
                            for lag in range(1, min(len(autocorr), 50)):
                                if autocorr[lag] > 0.05:
                                    tau_int += 2 * autocorr[lag]
                                else:
                                    break
                            
                            ess = len(data) / (2 * tau_int + 1)
                            ess_estimates.append(ess)
                    
                    mean_ess = np.mean(ess_estimates) if ess_estimates else 0
                else:
                    mean_ess = 0
                
                results[config["name"]] = {
                    "accept_rate": stats["acceptance_rate"],
                    "mean_hamiltonian_error": stats["mean_hamiltonian_error"],
                    "ess": mean_ess,
                    "samples": samples,
                    "success": True
                }
                
            except Exception as e:
                results[config["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        # 結果の分析と推奨事項
        print("設定別パフォーマンス:")
        print(f"{'設定':<20} {'受理率':<10} {'|ΔH|':<12} {'ESS':<10} {'状態':<10}")
        print("-" * 70)
        
        best_config = None
        best_score = -1
        
        for config_name, result in results.items():
            if result["success"]:
                accept_rate = result["accept_rate"]
                ham_error = result["mean_hamiltonian_error"]
                ess = result["ess"]
                
                # 総合スコア（受理率 * ESS / ハミルトニアン誤差）
                if ham_error > 0:
                    score = accept_rate * ess / np.log10(max(ham_error, 1e-10))
                else:
                    score = accept_rate * ess
                
                if score > best_score:
                    best_score = score
                    best_config = config_name
                
                print(f"{config_name:<20} {accept_rate:<10.3f} {ham_error:<12.6f} "
                      f"{ess:<10.1f} {'良好' if accept_rate > 0.5 else '要調整':<10}")
            else:
                print(f"{config_name:<20} {'---':<10} {'---':<12} {'---':<10} {'エラー':<10}")
        
        # 問題の特定と推奨事項
        print(f"\n推奨設定: {best_config}")
        
        # 問題診断
        problems = []
        if best_config:
            best_result = results[best_config]
            
            if best_result["accept_rate"] < 0.5:
                problems.append("受理率が低い")
            if best_result["mean_hamiltonian_error"] > 1.0:
                problems.append("ハミルトニアン誤差が大きい")
            if best_result["ess"] < 50:
                problems.append("有効サンプルサイズが小さい")
        
        if problems:
            print("検出された問題:")
            for problem in problems:
                print(f"  • {problem}")
            
            print("\n対処法:")
            if "受理率が低い" in problems:
                print("  • ステップサイズ εを小さくする")
                print("  • リープフロッグステップ数 L を減らす")
            if "ハミルトニアン誤差が大きい" in problems:
                print("  • ステップサイズ εを小さくする")
                print("  • 質量行列の調整を検討")
            if "有効サンプルサイズが小さい" in problems:
                print("  • リープフロッグステップ数 L を増やす")
                print("  • 前処理による変数変換を検討")
        else:
            print("問題は検出されませんでした。現在の設定が適切です。")
        
        return results
    
    # 問題のある分布でのテスト
    problematic_dists = create_problematic_distributions()
    
    all_results = {}
    for log_prob, grad_log_prob, name in problematic_dists:
        all_results[name] = diagnose_hmc_problems(log_prob, grad_log_prob, name)
    
    # 一般的な問題と解決策の表示
    print("\n" + "="*80)
    print("HMC一般的な問題と解決策")
    print("="*80)
    
    common_issues = {
        "低い受理率 (<50%)": {
            "原因": [
                "ステップサイズが大きすぎる",
                "リープフロッグステップ数が多すぎる",
                "分布の勾配が急峻"
            ],
            "解決策": [
                "εを小さくする (0.01-0.1程度)",
                "Lを減らす (5-20程度)",
                "質量行列による前処理",
                "変数変換による正規化"
            ]
        },
        
        "高い自己相関": {
            "原因": [
                "ステップサイズが小さすぎる",
                "軌跡が短すぎる",
                "分布に強い相関がある"
            ],
            "解決策": [
                "εを大きくする",
                "Lを増やす",
                "質量行列の適応",
                "再パラメータ化"
            ]
        },
        
        "数値不安定性": {
            "原因": [
                "勾配の計算誤差",
                "分布のスケールが異なる",
                "極値での数値オーバーフロー"
            ],
            "解決策": [
                "勾配の数値的検証",
                "スケーリングの正規化",
                "対数変換による安定化",
                "制約の適切な処理"
            ]
        },
        
        "マルチモーダル分布": {
            "原因": [
                "モード間の移動が困難",
                "エネルギー障壁が高い"
            ],
            "解決策": [
                "より長い軌跡 (L増加)",
                "テンパリング手法",
                "複数チェーンの並列実行",
                "変分推論との組み合わせ"
            ]
        }
    }
    
    for issue, details in common_issues.items():
        print(f"\n【{issue}】")
        print("原因:")
        for cause in details["原因"]:
            print(f"  • {cause}")
        print("解決策:")
        for solution in details["解決策"]:
            print(f"  • {solution}")
    
    # HMCチューニングのベストプラクティス
    print("\n" + "="*80)
    print("HMCチューニングのベストプラクティス")
    print("="*80)
    
    best_practices = [
        "1. まず標準設定(ε=0.1, L=20)から開始",
        "2. 受理率を60-80%の範囲に調整",
        "3. ハミルトニアン誤差をモニタリング",
        "4. 複数の初期値でテスト",
        "5. 収束診断を必ず実施",
        "6. 自己相関時間を確認",
        "7. 前処理（スケーリング）を検討",
        "8. 必要に応じて質量行列を適応",
        "9. 勾配の数値的検証を実施",
        "10. 実用ライブラリ(PyMC, Stan)の使用を検討"
    ]
    
    for practice in best_practices:
        print(f"  {practice}")
    
    return all_results

# HMCデバッグガイドの実行
debugging_results = hmc_debugging_guide()
```

---

## まとめ

このガイドでは、ハミルトニアンモンテカルロ法（HMC）の包括的な理解を目指して、以下の内容をカバーしました：

### 🎯 学習のポイント

1. **物理学的直感の重要性**
   - HMCは物理学の原理を統計計算に応用した画期的な手法
   - エネルギー保存と運動量の概念が効率的なサンプリングを実現

2. **アルゴリズムの段階的理解**
   - 基本概念から完全な実装まで段階的に構築
   - 各ステップの意味と重要性を詳細に解説

3. **実践的なスキル**
   - パラメータチューニングの体系的手法
   - 問題診断とデバッグの具体的手順
   - 実際の問題への応用例

4. **従来手法との比較**
   - ランダムウォークMHとの性能差の定量的分析
   - 高次元問題でのHMCの優位性

### 🚀 HMCの主な利点

- **効率的な探索**: 勾配情報を活用した賢い移動
- **高次元適応**: 次元数が増えても性能が劣化しにくい
- **理論的保証**: メトロポリス受理により正確なサンプリング
- **汎用性**: 勾配が計算できる任意の分布に適用可能

### ⚠️ 実用上の注意点

1. **パラメータ調整の重要性**
   - ステップサイズとリープフロッグステップ数の適切な設定
   - 分布の特性に応じたチューニングが必要

2. **計算コストの考慮**
   - 勾配計算のコストがボトルネックになる場合がある
   - 自動微分ライブラリの活用が重要

3. **数値安定性**
   - 条件数の高い分布での注意
   - 適切な前処理と変数変換の重要性

### 🔧 実践への移行

教育目的の実装から実用レベルへの移行には：

1. **専用ライブラリの活用**
   - PyMC, Stan, TensorFlow Probabilityなど
   - NUTS（No-U-Turn Sampler）の自動調整機能

2. **診断ツールの活用**
   - 収束診断とモニタリング
   - 複数チェーンによる検証

3. **モデリングスキルの向上**
   - 適切な事前分布の選択
   - 階層構造のモデリング
   - 制約の適切な処理

HMCは現代ベイズ統計学の中核的手法として、研究から実務まで幅広く活用されています。本ガイドで得た知識を基に、より高度なモデリングと推論に挑戦してください。

---

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "docs\u914d\u4e0b\u306e\u69cb\u9020\u3068HMC\u95a2\u9023\u8cc7\u6599\u3092\u78ba\u8a8d", "status": "completed", "priority": "high"}, {"id": "2", "content": "\u65e2\u5b58\u306eHMC\u30b3\u30f3\u30c6\u30f3\u30c4\uff08chapter6\uff09\u3092\u78ba\u8a8d", "status": "completed", "priority": "high"}, {"id": "3", "content": "HMC\u6559\u80b2\u30b3\u30f3\u30c6\u30f3\u30c4\u306e\u69cb\u6210\u3092\u8a2d\u8a08", "status": "completed", "priority": "medium"}, {"id": "4", "content": "HMC\u6559\u80b2\u30b3\u30f3\u30c6\u30f3\u30c4\u3092\u4f5c\u6210\u30fb\u5b9f\u88c5", "status": "completed", "priority": "medium"}]