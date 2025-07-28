# MCMCとは何か - 複雑な確率分布を制する計算革命

## 目次

1. [MCMCの本質と革命的意義](#第1章-mcmcの本質と革命的意義)
2. [数学的基礎理論](#第2章-数学的基礎理論)
3. [基本的なMCMCアルゴリズム](#第3章-基本的なmcmcアルゴリズム)
4. [実践的応用とワークフロー](#第4章-実践的応用とワークフロー)
5. [収束診断と性能評価](#第5章-収束診断と性能評価)
6. [高度なMCMC手法への展望](#第6章-高度なmcmc手法への展望)

---

## 第1章: MCMCの本質と革命的意義

### 1.1 計算困難な積分問題とMCMCの登場

現代の科学技術において、我々は日常的に複雑な確率分布と向き合っています。機械学習における予測の不確実性、統計物理学における粒子の挙動、金融工学におけるリスク評価、生物学における進化過程のモデリング。これらすべてに共通するのは、**「複雑な確率分布からサンプルを生成し、その性質を理解したい」**という根本的な要求です。

しかし、現実の問題で扱う確率分布は、しばしば以下のような困難を抱えています：

1. **高次元性**: 数十から数千のパラメータを持つ分布
2. **複雑な依存関係**: パラメータ間の非線形な相関
3. **正規化定数の未知性**: 分布の形は分かるが、積分値（正規化定数）が計算困難
4. **多峰性**: 複数のピークを持つ複雑な形状

従来の解析的手法では、このような分布から直接サンプルを得ることは不可能でした。そこで登場したのが**マルコフ連鎖モンテカルロ法（MCMC）**です。

MCMCは、「直接サンプリングできない複雑な分布を、間接的にマルコフ連鎖のシミュレーションによって攻略する」という革命的なアイデアです。この手法により、従来は理論上の存在でしかなかった複雑な確率モデルが、実際の計算機上で扱えるようになりました。

### 1.2 モンテカルロ法の基本原理

MCMCを理解するには、まず「モンテカルロ法」の考え方を理解する必要があります。モンテカルロ法とは、**乱数を用いた反復試行によって数値計算を行う手法**の総称です。

#### 円周率πの推定による理解

最も直感的な例として、モンテカルロ法による円周率πの推定を見てみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi_monte_carlo(n_points):
    """モンテカルロ法による円周率の推定"""
    # [-1, 1] x [-1, 1] の正方形内にランダムな点を生成
    points = np.random.uniform(-1, 1, (n_points, 2))
    
    # 原点からの距離が1以下の点（単位円内の点）を数える
    distances_squared = np.sum(points**2, axis=1)
    inside_circle = distances_squared <= 1
    
    # 円の面積 / 正方形の面積 = π/4
    pi_estimate = 4 * np.sum(inside_circle) / n_points
    
    return pi_estimate, points, inside_circle

# 異なるサンプル数での推定
sample_sizes = [100, 1000, 10000, 100000]
pi_estimates = []

for n in sample_sizes:
    pi_est, _, _ = estimate_pi_monte_carlo(n)
    pi_estimates.append(pi_est)
    print(f"n={n:6d}: π ≈ {pi_est:.6f}, 誤差 = {abs(pi_est - np.pi):.6f}")

# 可視化
n_vis = 5000
pi_est, points, inside_circle = estimate_pi_monte_carlo(n_vis)

plt.figure(figsize=(8, 8))
plt.scatter(points[inside_circle, 0], points[inside_circle, 1], 
           c='blue', s=1, alpha=0.6, label=f'円内 ({np.sum(inside_circle)}点)')
plt.scatter(points[~inside_circle, 0], points[~inside_circle, 1], 
           c='red', s=1, alpha=0.6, label=f'円外 ({np.sum(~inside_circle)}点)')

# 単位円を描画
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f'モンテカルロ法による円周率推定\n推定値: π ≈ {pi_est:.6f} (真値: {np.pi:.6f})')
plt.show()
```

このシミュレーションから分かる重要なポイント：

1. **大数の法則**: サンプル数を増やすほど、推定精度が向上する
2. **確率的収束**: 各試行では異なる結果が得られるが、平均的には真値に収束
3. **単純な仕組み**: 複雑な計算（円の面積）を、単純な確率実験（点の内外判定）に置き換え

この「複雑な計算を単純な確率実験に置き換える」というアイデアこそ、モンテカルロ法の本質です。

### 1.3 マルコフ連鎖と定常分布の概念

次に、MCMCのもう一つの柱である「マルコフ連鎖」について学びましょう。

#### マルコフ性（無記憶性）

マルコフ連鎖とは、**「次の状態が現在の状態のみに依存し、過去の履歴には依存しない」**という性質（マルコフ性）を持つ確率過程です。

数学的には：
$$P(X_{t+1} = j | X_t = i, X_{t-1} = i_{t-1}, \ldots, X_0 = i_0) = P(X_{t+1} = j | X_t = i)$$

#### 天気予報モデルによる直感的理解

簡単な例として、3状態（晴れ、曇り、雨）の天気モデルを考えてみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

# 遷移確率行列の定義
# 行: 今日の天気, 列: 明日の天気
# 0: 晴れ, 1: 曇り, 2: 雨
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # 晴れ → [晴れ, 曇り, 雨]
    [0.3, 0.4, 0.3],  # 曇り → [晴れ, 曇り, 雨]
    [0.2, 0.3, 0.5]   # 雨 → [晴れ, 曇り, 雨]
])

weather_names = ['晴れ', '曇り', '雨']

def simulate_weather_chain(initial_state, n_days, transition_matrix):
    """天気のマルコフ連鎖をシミュレーション"""
    states = [initial_state]
    current_state = initial_state
    
    for day in range(n_days - 1):
        # 現在の状態から次の状態への遷移確率を取得
        prob_dist = transition_matrix[current_state]
        # 次の状態をサンプリング
        next_state = np.random.choice(3, p=prob_dist)
        states.append(next_state)
        current_state = next_state
    
    return states

# 定常分布の理論値を計算
# π = π * P の固有ベクトルを求める
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
stationary_idx = np.argmax(eigenvalues.real)
stationary_dist = eigenvectors[:, stationary_idx].real
stationary_dist = stationary_dist / np.sum(stationary_dist)

print("遷移確率行列:")
print(transition_matrix)
print(f"\n理論的定常分布: {dict(zip(weather_names, stationary_dist))}")

# 長期シミュレーション
n_days = 10000
weather_chain = simulate_weather_chain(0, n_days, transition_matrix)

# 経験的分布の計算（後半のデータのみ使用）
burn_in = 1000
empirical_dist = np.bincount(weather_chain[burn_in:]) / (n_days - burn_in)

print(f"経験的分布（{burn_in}日後以降）: {dict(zip(weather_names, empirical_dist))}")

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 時系列プロット
days = range(min(100, n_days))
ax1.plot(days, weather_chain[:len(days)], 'o-', markersize=3, alpha=0.7)
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(weather_names)
ax1.set_xlabel('日数')
ax1.set_ylabel('天気')
ax1.set_title('天気の時系列変化（最初の100日）')
ax1.grid(True, alpha=0.3)

# 分布の比較
x = np.arange(3)
width = 0.35
ax2.bar(x - width/2, stationary_dist, width, label='理論的定常分布', alpha=0.7)
ax2.bar(x + width/2, empirical_dist, width, label='経験的分布', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(weather_names)
ax2.set_ylabel('確率')
ax2.set_title('定常分布の比較')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

この例から重要な概念が見えてきます：

1. **定常分布**: 長期的に見ると、天気の出現頻度が一定の分布に収束する
2. **初期値忘却**: 最初の天気が何であれ、十分時間が経てば同じ分布に収束する
3. **エルゴード性**: 時間平均が確率平均と一致する

### 1.4 MCMCが解決する現実問題の範囲

MCMCは単なる数学的手法ではなく、現代科学技術の幅広い分野で実際の問題を解決しています。

#### 1.4.1 ベイズ統計・機械学習

**問題**: 複雑なモデルの事後分布からサンプリング

```python
# ベイズ線形回帰の例（概念的なコード）
def bayesian_linear_regression_mcmc(X, y, n_samples=10000):
    """
    ベイズ線形回帰のMCMC実装
    事後分布 p(β, σ² | X, y) からサンプリング
    """
    # 事前分布: β ~ N(0, I), σ² ~ InvGamma(a, b)
    beta_samples = []
    sigma2_samples = []
    
    # 初期値
    beta = np.zeros(X.shape[1])
    sigma2 = 1.0
    
    for i in range(n_samples):
        # ギブスサンプリング
        # p(β | σ², X, y) からサンプリング
        beta = sample_beta_given_sigma2(X, y, sigma2)
        
        # p(σ² | β, X, y) からサンプリング  
        sigma2 = sample_sigma2_given_beta(X, y, beta)
        
        beta_samples.append(beta.copy())
        sigma2_samples.append(sigma2)
    
    return np.array(beta_samples), np.array(sigma2_samples)
```

#### 1.4.2 統計物理学

**問題**: イジングモデルの配置サンプリング

統計物理学では、系の状態（スピン配置など）の確率分布は通常、ボルツマン分布に従います：

$$p(\mathbf{s}) = \frac{1}{Z} \exp(-\beta H(\mathbf{s}))$$

ここで、$Z$は分配関数（正規化定数）で、直接計算することは困難です。MCMCを使うことで、この分布からサンプルを生成し、系の物理的性質を調べることができます。

#### 1.4.3 金融工学

**問題**: ポートフォリオ最適化とリスク評価

金融市場では、資産価格の確率分布は複雑で非ガウス的です。MCMCを使って：
- 極値理論に基づくVaR（Value at Risk）の計算
- 信用リスクモデルのパラメータ推定
- 複雑なデリバティブの価格評価

#### 1.4.4 計算生物学

**問題**: 系統樹推定とタンパク質構造予測

```python
# 系統樹のMCMCサンプリング（概念的）
def phylogenetic_mcmc(sequences, n_iterations=100000):
    """
    分子系統樹のMCMC推定
    事後分布 p(樹構造, 枝長 | 配列データ) からサンプリング
    """
    current_tree = initialize_random_tree(sequences)
    
    for iteration in range(n_iterations):
        # 提案: 樹構造または枝長を変更
        proposed_tree = propose_tree_change(current_tree)
        
        # 受容確率の計算
        likelihood_ratio = compute_likelihood_ratio(proposed_tree, current_tree, sequences)
        prior_ratio = compute_prior_ratio(proposed_tree, current_tree)
        
        acceptance_prob = min(1, likelihood_ratio * prior_ratio)
        
        if np.random.random() < acceptance_prob:
            current_tree = proposed_tree
        
        if iteration % 1000 == 0:
            save_tree_sample(current_tree)
    
    return tree_samples
```

#### 1.4.5 画像処理・コンピュータビジョン

**問題**: 画像のノイズ除去やセグメンテーション

マルコフ確率場（MRF）を使った画像処理では、各ピクセルの値が近隣ピクセルと相関を持ちます。この複雑な依存構造を持つ分布からサンプリングすることで、画像復元や物体認識が可能になります。

### 1.5 MCMCの革命的意義

MCMCが科学技術に与えた影響は計り知れません：

1. **理論と実践の橋渡し**: 数学的に美しいが計算困難なモデルを実用化
2. **不確実性の定量化**: 点推定だけでなく、予測の信頼区間も提供
3. **複雑系の理解**: 高次元・非線形・多峰性を持つ系の解析を可能に
4. **学際的応用**: 物理学、生物学、経済学、工学など分野横断的な手法として普及

現在、MCMCを使わずに現代的なデータサイエンスや科学計算を行うことは困難と言っても過言ではありません。この手法こそが、21世紀の「計算による科学」を支える重要な基盤技術なのです。

次章では、この強力な手法の数学的基礎を詳しく見ていきます。

---

## 第2章: 数学的基礎理論

### 2.1 詳細釣り合い条件とエルゴード性

MCMCの理論的基盤を理解するには、「なぜマルコフ連鎖が目標分布に収束するのか」という根本的な疑問に答える必要があります。その鍵となるのが**詳細釣り合い条件**です。

#### 詳細釣り合い条件（Detailed Balance Condition）

マルコフ連鎖が目標分布$\pi(x)$を定常分布として持つための十分条件は、以下の詳細釣り合い条件を満たすことです：

$$\pi(x) P(x \to y) = \pi(y) P(y \to x)$$

ここで、$P(x \to y)$は状態$x$から状態$y$への遷移確率です。

#### 直感的理解：人口移動の例

この条件を理解するために、2つの都市A、Bの間の人口移動を考えてみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

def population_migration_example():
    """人口移動による詳細釣り合い条件の理解"""
    
    # 都市の定常人口（目標分布）
    pop_A_stationary = 0.7  # 都市Aの人口比率
    pop_B_stationary = 0.3  # 都市Bの人口比率
    
    # 移住率（年間）
    migration_rate_A_to_B = 0.1  # AからBへの移住率
    
    # 詳細釣り合い条件から、BからAへの移住率を計算
    # π(A) * P(A→B) = π(B) * P(B→A)
    # 0.7 * 0.1 = 0.3 * P(B→A)
    migration_rate_B_to_A = (pop_A_stationary * migration_rate_A_to_B) / pop_B_stationary
    
    print(f"都市Aの定常人口比率: {pop_A_stationary}")
    print(f"都市Bの定常人口比率: {pop_B_stationary}")
    print(f"A→Bの移住率: {migration_rate_A_to_B}")
    print(f"B→Aの移住率: {migration_rate_B_to_A:.3f}")
    
    # 人口フローの計算
    flow_A_to_B = pop_A_stationary * migration_rate_A_to_B
    flow_B_to_A = pop_B_stationary * migration_rate_B_to_A
    
    print(f"\n年間人口フロー:")
    print(f"A→B: {flow_A_to_B:.3f}")
    print(f"B→A: {flow_B_to_A:.3f}")
    print(f"フローの差: {abs(flow_A_to_B - flow_B_to_A):.6f}")
    
    # シミュレーションによる検証
    n_years = 100
    initial_pop_A = 0.5  # 初期状態（平衡状態から離れた状態）
    
    pop_A_history = [initial_pop_A]
    current_pop_A = initial_pop_A
    
    for year in range(n_years):
        current_pop_B = 1 - current_pop_A
        
        # 移住による人口変化
        move_A_to_B = current_pop_A * migration_rate_A_to_B
        move_B_to_A = current_pop_B * migration_rate_B_to_A
        
        current_pop_A = current_pop_A - move_A_to_B + move_B_to_A
        pop_A_history.append(current_pop_A)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    years = range(n_years + 1)
    plt.plot(years, pop_A_history, 'b-', linewidth=2, label='都市Aの人口比率')
    plt.axhline(y=pop_A_stationary, color='r', linestyle='--', 
                label=f'定常状態 ({pop_A_stationary})')
    plt.xlabel('年数')
    plt.ylabel('都市Aの人口比率')
    plt.title('人口移動による定常状態への収束')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pop_A_history

# 実行
population_history = population_migration_example()
```

このシミュレーションから分かること：

1. **局所的な釣り合い**: 各都市ペア間で流入と流出が釣り合っている
2. **大域的な安定性**: 局所的な釣り合いが大域的な定常状態を保証
3. **初期値忘却**: 初期の人口分布によらず、同じ定常状態に収束

#### エルゴード性

マルコフ連鎖が確実に定常分布に収束するためには、**エルゴード性**が必要です。エルゴード性は主に以下の条件から構成されます：

1. **既約性（Irreducibility）**: どの状態からも他のどの状態にも有限ステップで到達可能
2. **非周期性（Aperiodicity）**: 状態の遷移に固定的な周期がない

```python
def ergodicity_demonstration():
    """エルゴード性の視覚的デモンストレーション"""
    
    # 既約なマルコフ連鎖の例
    irreducible_matrix = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2], 
        [0.3, 0.3, 0.4]
    ])
    
    # 可約なマルコフ連鎖の例（状態0からは状態2に到達不可能）
    reducible_matrix = np.array([
        [0.7, 0.3, 0.0],
        [0.4, 0.6, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    def simulate_chain(transition_matrix, initial_state, n_steps):
        states = [initial_state]
        current_state = initial_state
        
        for _ in range(n_steps):
            prob_dist = transition_matrix[current_state]
            next_state = np.random.choice(len(prob_dist), p=prob_dist)
            states.append(next_state)
            current_state = next_state
        
        return states
    
    n_steps = 1000
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 既約チェーン
    for i, initial_state in enumerate([0, 2]):
        states = simulate_chain(irreducible_matrix, initial_state, n_steps)
        axes[0, i].plot(states[:200], alpha=0.7)
        axes[0, i].set_title(f'既約チェーン (初期状態: {initial_state})')
        axes[0, i].set_ylabel('状態')
        axes[0, i].set_ylim(-0.5, 2.5)
        axes[0, i].grid(True, alpha=0.3)
    
    # 可約チェーン  
    for i, initial_state in enumerate([0, 2]):
        states = simulate_chain(reducible_matrix, initial_state, n_steps)
        axes[1, i].plot(states[:200], alpha=0.7, color='red')
        axes[1, i].set_title(f'可約チェーン (初期状態: {initial_state})')
        axes[1, i].set_xlabel('ステップ')
        axes[1, i].set_ylabel('状態')
        axes[1, i].set_ylim(-0.5, 2.5)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 実行
ergodicity_demonstration()
```

### 2.2 収束理論と定常分布への到達

#### 収束の理論的保証

エルゴード的なマルコフ連鎖について、以下の重要な定理が成り立ちます：

**エルゴード定理**: エルゴード的なマルコフ連鎖$\{X_t\}$に対して、初期分布によらず
$$\lim_{n \to \infty} \frac{1}{n} \sum_{t=1}^{n} f(X_t) = \mathbb{E}_\pi[f(X)] \quad \text{確率1で}$$

これは、**時間平均が確率平均と一致する**ことを意味し、MCMCサンプリングの理論的正当性を保証します。

#### 収束速度の評価

実際の応用では、「どのくらいの時間で定常分布に近づくか」が重要です。収束速度は、遷移確率行列の**第二固有値**と密接に関係しています。

```python
def convergence_rate_analysis():
    """収束速度の分析"""
    
    # 異なる収束速度を持つマルコフ連鎖の比較
    
    # 速い収束（弱い相関）
    fast_matrix = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    
    # 遅い収束（強い相関）
    slow_matrix = np.array([
        [0.95, 0.05],
        [0.1, 0.9]
    ])
    
    def analyze_convergence(transition_matrix, name):
        # 固有値を計算
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        
        # 第二固有値（収束速度を決定）
        sorted_eigenvals = np.sort(np.abs(eigenvalues))[::-1]
        second_eigenval = sorted_eigenvals[1]
        
        print(f"{name}:")
        print(f"  第二固有値: {second_eigenval:.4f}")
        print(f"  収束率: {-np.log(second_eigenval):.4f}")
        
        # 定常分布
        stationary_idx = np.argmax(np.abs(eigenvalues))
        stationary_dist = np.abs(eigenvectors[:, stationary_idx])
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        print(f"  定常分布: [{stationary_dist[0]:.3f}, {stationary_dist[1]:.3f}]")
        
        return second_eigenval, stationary_dist
    
    # 解析
    fast_eigen, fast_stationary = analyze_convergence(fast_matrix, "高速収束チェーン")
    slow_eigen, slow_stationary = analyze_convergence(slow_matrix, "低速収束チェーン")
    
    # シミュレーションによる検証
    def simulate_convergence(transition_matrix, stationary_dist, n_steps=1000):
        current_dist = np.array([1.0, 0.0])  # 初期分布
        dist_history = [current_dist.copy()]
        
        for _ in range(n_steps):
            current_dist = current_dist @ transition_matrix
            dist_history.append(current_dist.copy())
        
        # 定常分布からの距離
        distances = [np.linalg.norm(dist - stationary_dist) for dist in dist_history]
        return distances
    
    fast_distances = simulate_convergence(fast_matrix, fast_stationary)
    slow_distances = simulate_convergence(slow_matrix, slow_stationary)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    steps = range(len(fast_distances))
    plt.semilogy(steps[:100], fast_distances[:100], 'b-', label='高速収束')
    plt.semilogy(steps[:100], slow_distances[:100], 'r-', label='低速収束')
    plt.xlabel('ステップ')
    plt.ylabel('定常分布からの距離（対数スケール）')
    plt.title('収束速度の比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 自己相関の理論値
    lags = np.arange(0, 50)
    fast_autocorr = fast_eigen ** lags
    slow_autocorr = slow_eigen ** lags
    
    plt.plot(lags, fast_autocorr, 'b-', label='高速収束')
    plt.plot(lags, slow_autocorr, 'r-', label='低速収束')
    plt.xlabel('ラグ')
    plt.ylabel('自己相関')
    plt.title('理論的自己相関関数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 実行
convergence_rate_analysis()
```

### 2.3 MCMCの理論的保証

#### 中心極限定理

MCMCサンプル$\{X_1, X_2, \ldots, X_n\}$に対して、適切な条件下で以下の中心極限定理が成り立ちます：

$$\sqrt{n}\left(\frac{1}{n}\sum_{i=1}^{n} f(X_i) - \mathbb{E}_\pi[f(X)]\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2_{\text{MCMC}})$$

ここで、$\sigma^2_{\text{MCMC}}$は**MCMC標準誤差**で、サンプル間の相関を考慮した分散です：

$$\sigma^2_{\text{MCMC}} = \text{Var}_\pi[f(X)] \left(1 + 2\sum_{k=1}^{\infty} \rho_k\right)$$

$\rho_k$は自己相関関数です。

#### 実効サンプルサイズ

独立なサンプルとの比較において、MCMCサンプルの「実質的な」サンプルサイズは：

$$n_{\text{eff}} = \frac{n}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

で与えられます。

```python
def effective_sample_size_demo():
    """実効サンプルサイズのデモンストレーション"""
    
    def autocorr_function(x, max_lag=50):
        """自己相関関数の計算"""
        n = len(x)
        x = x - np.mean(x)
        autocorrs = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorrs.append(1.0)
            else:
                c_lag = np.mean(x[:-lag] * x[lag:])
                c_0 = np.mean(x**2)
                autocorrs.append(c_lag / c_0)
                
        return np.array(autocorrs)
    
    def effective_sample_size(x):
        """実効サンプルサイズの計算"""
        autocorrs = autocorr_function(x)
        
        # 自己相関が負になる最初のポイントで打ち切り
        cutoff = len(autocorrs)
        for i in range(1, len(autocorrs)):
            if autocorrs[i] <= 0:
                cutoff = i
                break
        
        # 実効サンプルサイズ
        tau_int = 1 + 2 * np.sum(autocorrs[1:cutoff])
        n_eff = len(x) / tau_int
        
        return n_eff, tau_int, autocorrs
    
    # 異なる相関を持つサンプルの生成
    np.random.seed(42)
    n_samples = 10000
    
    # 独立サンプル
    independent_samples = np.random.normal(0, 1, n_samples)
    
    # 相関を持つサンプル（AR(1)プロセス）
    rho = 0.8
    correlated_samples = np.zeros(n_samples)
    correlated_samples[0] = np.random.normal()
    for i in range(1, n_samples):
        correlated_samples[i] = rho * correlated_samples[i-1] + np.sqrt(1 - rho**2) * np.random.normal()
    
    # 実効サンプルサイズの計算
    n_eff_indep, tau_int_indep, autocorr_indep = effective_sample_size(independent_samples)
    n_eff_corr, tau_int_corr, autocorr_corr = effective_sample_size(correlated_samples)
    
    print(f"独立サンプル:")
    print(f"  総サンプル数: {n_samples}")
    print(f"  実効サンプルサイズ: {n_eff_indep:.0f}")
    print(f"  積分自己相関時間: {tau_int_indep:.2f}")
    
    print(f"\n相関サンプル (ρ={rho}):")
    print(f"  総サンプル数: {n_samples}")
    print(f"  実効サンプルサイズ: {n_eff_corr:.0f}")
    print(f"  積分自己相関時間: {tau_int_corr:.2f}")
    print(f"  効率: {n_eff_corr/n_samples:.1%}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # トレースプロット
    axes[0, 0].plot(independent_samples[:1000], alpha=0.7, label='独立')
    axes[0, 0].set_title('独立サンプルのトレース')
    axes[0, 0].set_ylabel('値')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(correlated_samples[:1000], alpha=0.7, label='相関', color='red')
    axes[0, 1].set_title(f'相関サンプルのトレース (ρ={rho})')
    axes[0, 1].set_ylabel('値')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 自己相関関数
    lags = range(len(autocorr_indep))
    axes[1, 0].plot(lags[:50], autocorr_indep[:50], 'b-', label='独立')
    axes[1, 0].plot(lags[:50], autocorr_corr[:50], 'r-', label='相関')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('ラグ')
    axes[1, 0].set_ylabel('自己相関')
    axes[1, 0].set_title('自己相関関数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 実効サンプルサイズの比較
    categories = ['独立', '相関']
    n_effs = [n_eff_indep, n_eff_corr]
    
    axes[1, 1].bar(categories, n_effs, color=['blue', 'red'], alpha=0.7)
    axes[1, 1].axhline(n_samples, color='k', linestyle='--', alpha=0.5, label='総サンプル数')
    axes[1, 1].set_ylabel('実効サンプルサイズ')
    axes[1, 1].set_title('実効サンプルサイズの比較')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 実行
effective_sample_size_demo()
```

### 2.4 実装に必要な数学的準備

#### 受容確率の導出

メトロポリス・ヘイスティングス法の受容確率は、詳細釣り合い条件から導出されます。

現在の状態を$x$、提案された状態を$y$とすると：
$$\pi(x) P(x \to y) = \pi(y) P(y \to x)$$

遷移確率を提案確率$q(y|x)$と受容確率$\alpha(y|x)$の積で表すと：
$$P(x \to y) = q(y|x) \alpha(y|x)$$

詳細釣り合い条件に代入すると：
$$\pi(x) q(y|x) \alpha(y|x) = \pi(y) q(x|y) \alpha(x|y)$$

これより、メトロポリス・ヘイスティングス受容確率が導出されます：
$$\alpha(y|x) = \min\left(1, \frac{\pi(y) q(x|y)}{\pi(x) q(y|x)}\right)$$

この数学的基盤の上に、次章で具体的なアルゴリズムを構築していきます。

## 第3章: 基本的なMCMCアルゴリズム

### 3.1 メトロポリス法

#### 3.1.1 アルゴリズムの原理

メトロポリス法は、MCMCアルゴリズムの中で最も基本的かつ直感的な手法です。**対称な提案分布**を用いて、目標分布からのサンプリングを行います。

**アルゴリズムの手順**：

1. **初期化**: 初期値$x^{(0)}$を設定
2. **反復** ($t = 0, 1, 2, \ldots$):
   - **提案**: 対称な提案分布$q(x'|x^{(t)})$から候補$x'$をサンプル
   - **受容確率の計算**: $\alpha = \min\left(1, \frac{\pi(x')}{\pi(x^{(t)})}\right)$
   - **受容/棄却判定**: 
     - 確率$\alpha$で$x^{(t+1)} = x'$（受容）
     - 確率$(1-\alpha)$で$x^{(t+1)} = x^{(t)}$（棄却）

#### 3.1.2 完全実装例：正規分布からのサンプリング

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class MetropolisSampler:
    def __init__(self, target_logpdf, proposal_std=1.0):
        """
        メトロポリス法のサンプラー
        
        Parameters:
        -----------
        target_logpdf : callable
            目標分布の対数確率密度関数
        proposal_std : float
            提案分布（正規分布）の標準偏差
        """
        self.target_logpdf = target_logpdf
        self.proposal_std = proposal_std
        self.samples = []
        self.n_accepted = 0
        self.n_proposed = 0
    
    def sample(self, n_samples, initial_value=0.0, burn_in=1000):
        """サンプリングの実行"""
        current_x = initial_value
        current_logp = self.target_logpdf(current_x)
        
        # サンプルを保存するリスト
        samples = []
        
        # バーンイン + 本サンプリング
        total_samples = burn_in + n_samples
        
        for i in range(total_samples):
            # 提案（現在の位置を中心とした正規分布）
            proposal_x = current_x + np.random.normal(0, self.proposal_std)
            proposal_logp = self.target_logpdf(proposal_x)
            
            # 受容確率の計算（対数スケールで計算して数値安定性を向上）
            log_alpha = min(0, proposal_logp - current_logp)
            alpha = np.exp(log_alpha)
            
            # 受容/棄却判定
            if np.random.random() < alpha:
                current_x = proposal_x
                current_logp = proposal_logp
                self.n_accepted += 1
            
            self.n_proposed += 1
            
            # バーンイン後のサンプルのみ保存
            if i >= burn_in:
                samples.append(current_x)
        
        self.samples = np.array(samples)
        return self.samples
    
    def acceptance_rate(self):
        """受容率の計算"""
        return self.n_accepted / self.n_proposed if self.n_proposed > 0 else 0.0
    
    def diagnostic_plots(self, true_samples=None):
        """診断プロットの作成"""
        if len(self.samples) == 0:
            print("サンプルが生成されていません。")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # トレースプロット
        axes[0, 0].plot(self.samples[:2000])
        axes[0, 0].set_title('トレースプロット（最初の2000サンプル）')
        axes[0, 0].set_xlabel('イテレーション')
        axes[0, 0].set_ylabel('値')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ヒストグラム vs 真の分布
        axes[0, 1].hist(self.samples, bins=50, density=True, alpha=0.7, 
                       label='MCMCサンプル')
        if true_samples is not None:
            axes[0, 1].hist(true_samples, bins=50, density=True, alpha=0.5, 
                           label='真の分布からのサンプル')
        axes[0, 1].set_title('サンプル分布の比較')
        axes[0, 1].set_xlabel('値')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 自己相関関数
        def autocorr(x, max_lag=50):
            x = x - np.mean(x)
            autocorrs = np.correlate(x, x, mode='full')
            autocorrs = autocorrs[autocorrs.size // 2:]
            autocorrs = autocorrs / autocorrs[0]
            return autocorrs[:max_lag]
        
        lags = range(50)
        autocorrs = autocorr(self.samples)
        axes[1, 0].plot(lags, autocorrs)
        axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('自己相関関数')
        axes[1, 0].set_xlabel('ラグ')
        axes[1, 0].set_ylabel('自己相関')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累積平均の収束
        cumulative_mean = np.cumsum(self.samples) / np.arange(1, len(self.samples) + 1)
        axes[1, 1].plot(cumulative_mean)
        if true_samples is not None:
            true_mean = np.mean(true_samples)
            axes[1, 1].axhline(true_mean, color='r', linestyle='--', 
                              label=f'真の平均: {true_mean:.3f}')
            axes[1, 1].legend()
        axes[1, 1].set_title('累積平均の収束')
        axes[1, 1].set_xlabel('イテレーション')
        axes[1, 1].set_ylabel('累積平均')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 統計要約
        print(f"受容率: {self.acceptance_rate():.3f}")
        print(f"サンプル数: {len(self.samples)}")
        print(f"平均: {np.mean(self.samples):.3f}")
        print(f"標準偏差: {np.std(self.samples):.3f}")

# 使用例：標準正規分布からのサンプリング
def standard_normal_logpdf(x):
    """標準正規分布の対数確率密度関数"""
    return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

# サンプラーの作成と実行
sampler = MetropolisSampler(standard_normal_logpdf, proposal_std=1.0)
mcmc_samples = sampler.sample(n_samples=10000, initial_value=0.0)

# 比較用の真のサンプル
true_samples = np.random.normal(0, 1, 10000)

# 診断プロット
sampler.diagnostic_plots(true_samples)
```

#### 3.1.3 提案分布の調整とチューニング

提案分布の標準偏差は、サンプリング効率に大きく影響します。最適な値を見つけるためのチューニング例を見てみましょう。

```python
def proposal_tuning_experiment():
    """提案分布のチューニング実験"""
    
    # 異なる提案標準偏差での実験
    proposal_stds = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for std in proposal_stds:
        sampler = MetropolisSampler(standard_normal_logpdf, proposal_std=std)
        samples = sampler.sample(n_samples=5000, burn_in=1000)
        
        results[std] = {
            'acceptance_rate': sampler.acceptance_rate(),
            'samples': samples,
            'effective_sample_size': len(samples) / (1 + 2 * np.sum(
                [np.corrcoef(samples[:-lag], samples[lag:])[0,1] 
                 for lag in range(1, min(50, len(samples)//4)) 
                 if not np.isnan(np.corrcoef(samples[:-lag], samples[lag:])[0,1])]
            ))
        }
    
    # 結果の可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 受容率 vs 提案標準偏差
    stds = list(results.keys())
    acceptance_rates = [results[std]['acceptance_rate'] for std in stds]
    ess_values = [results[std]['effective_sample_size'] for std in stds]
    
    axes[0, 0].plot(stds, acceptance_rates, 'o-')
    axes[0, 0].axhline(0.44, color='r', linestyle='--', label='理論最適値')
    axes[0, 0].set_xlabel('提案標準偏差')
    axes[0, 0].set_ylabel('受容率')
    axes[0, 0].set_title('受容率 vs 提案標準偏差')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(stds, ess_values, 'o-', color='green')
    axes[0, 1].set_xlabel('提案標準偏差')
    axes[0, 1].set_ylabel('実効サンプルサイズ')
    axes[0, 1].set_title('効率性 vs 提案標準偏差')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 各提案標準偏差でのトレースプロット
    for i, std in enumerate([0.1, 1.0, 5.0]):
        row = (i + 2) // 3
        col = (i + 2) % 3
        
        samples = results[std]['samples'][:1000]
        axes[row, col].plot(samples)
        axes[row, col].set_title(f'std={std}, 受容率={results[std]["acceptance_rate"]:.3f}')
        axes[row, col].set_ylabel('値')
        if row == 1:
            axes[row, col].set_xlabel('イテレーション')
        axes[row, col].grid(True, alpha=0.3)
    
    # 結果表示
    axes[1, 2].axis('off')
    result_text = "提案標準偏差別結果:\n\n"
    for std in stds:
        result_text += f"std={std}:\n"
        result_text += f"  受容率: {results[std]['acceptance_rate']:.3f}\n"
        result_text += f"  ESS: {results[std]['effective_sample_size']:.0f}\n\n"
    
    axes[1, 2].text(0.1, 0.9, result_text, transform=axes[1, 2].transAxes, 
                   verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return results

# 実験実行
tuning_results = proposal_tuning_experiment()
```

### 3.2 メトロポリス・ヘイスティングス法

#### 3.2.1 一般化された受容確率

メトロポリス・ヘイスティングス法は、非対称な提案分布にも対応できるメトロポリス法の一般化です。

受容確率は以下で与えられます：
$$\alpha(x'|x) = \min\left(1, \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}\right)$$

#### 3.2.2 実装例：非対称提案分布を用いたサンプリング

```python
class MetropolisHastingsSampler:
    def __init__(self, target_logpdf, proposal_sampler, proposal_logpdf):
        """
        メトロポリス・ヘイスティングス法のサンプラー
        
        Parameters:
        -----------
        target_logpdf : callable
            目標分布の対数確率密度関数
        proposal_sampler : callable
            提案分布からのサンプリング関数 (current_x) -> proposal_x
        proposal_logpdf : callable
            提案分布の対数確率密度関数 (proposal_x, current_x) -> logpdf
        """
        self.target_logpdf = target_logpdf
        self.proposal_sampler = proposal_sampler
        self.proposal_logpdf = proposal_logpdf
        self.samples = []
        self.n_accepted = 0
        self.n_proposed = 0
    
    def sample(self, n_samples, initial_value=0.0, burn_in=1000):
        """サンプリングの実行"""
        current_x = initial_value
        current_logp = self.target_logpdf(current_x)
        
        samples = []
        total_samples = burn_in + n_samples
        
        for i in range(total_samples):
            # 提案
            proposal_x = self.proposal_sampler(current_x)
            proposal_logp = self.target_logpdf(proposal_x)
            
            # 受容確率の計算
            log_alpha = (proposal_logp + self.proposal_logpdf(current_x, proposal_x) - 
                        current_logp - self.proposal_logpdf(proposal_x, current_x))
            log_alpha = min(0, log_alpha)
            alpha = np.exp(log_alpha)
            
            # 受容/棄却判定
            if np.random.random() < alpha:
                current_x = proposal_x
                current_logp = proposal_logp
                self.n_accepted += 1
            
            self.n_proposed += 1
            
            if i >= burn_in:
                samples.append(current_x)
        
        self.samples = np.array(samples)
        return self.samples
    
    def acceptance_rate(self):
        return self.n_accepted / self.n_proposed if self.n_proposed > 0 else 0.0

# 例：指数分布を提案分布として使用
def exponential_proposal_example():
    """指数分布を提案分布とした例"""
    
    # 目標分布：ガンマ分布 Gamma(2, 1)
    def gamma_logpdf(x):
        if x <= 0:
            return -np.inf
        return (2-1) * np.log(x) - x - np.log(1)  # Gamma(2,1)の対数pdf
    
    # 提案分布：指数分布 Exp(1/current_x)
    def exp_proposal_sampler(current_x):
        rate = 1.0 / max(current_x, 0.1)  # 数値安定性のための下限
        return np.random.exponential(1.0 / rate)
    
    def exp_proposal_logpdf(proposal_x, current_x):
        if proposal_x <= 0:
            return -np.inf
        rate = 1.0 / max(current_x, 0.1)
        return np.log(rate) - rate * proposal_x
    
    # サンプリング実行
    mh_sampler = MetropolisHastingsSampler(
        target_logpdf=gamma_logpdf,
        proposal_sampler=exp_proposal_sampler,
        proposal_logpdf=exp_proposal_logpdf
    )
    
    mh_samples = mh_sampler.sample(n_samples=10000, initial_value=1.0)
    
    # 真のガンマ分布からのサンプル
    true_samples = np.random.gamma(2, 1, 10000)
    
    # 結果の比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ヒストグラム比較
    axes[0].hist(mh_samples, bins=50, density=True, alpha=0.7, label='MH法')
    axes[0].hist(true_samples, bins=50, density=True, alpha=0.5, label='真の分布')
    axes[0].set_xlabel('値')
    axes[0].set_ylabel('密度')
    axes[0].set_title('分布の比較')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Qプロット
    from scipy import stats
    stats.probplot(mh_samples, dist=stats.gamma(2, scale=1), plot=axes[1])
    axes[1].set_title('Q-Qプロット（ガンマ分布）')
    axes[1].grid(True, alpha=0.3)
    
    # トレースプロット
    axes[2].plot(mh_samples[:2000])
    axes[2].set_xlabel('イテレーション')
    axes[2].set_ylabel('値')
    axes[2].set_title(f'トレースプロット（受容率: {mh_sampler.acceptance_rate():.3f}）')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"受容率: {mh_sampler.acceptance_rate():.3f}")
    print(f"MH法平均: {np.mean(mh_samples):.3f}")
    print(f"真の平均: {np.mean(true_samples):.3f}")
    print(f"理論平均: {2.0}")

# 実行
exponential_proposal_example()
```

### 3.3 ギブスサンプリング

#### 3.3.1 条件付き分布を利用したサンプリング

ギブスサンプリングは、多変量分布において各変数を条件付き分布から順次サンプリングする手法です。特に**共役事前分布**を持つベイズモデルで威力を発揮します。

#### 3.3.2 2変量正規分布からのギブスサンプリング

```python
class GibbsSampler2D:
    def __init__(self, mu, Sigma):
        """
        2次元正規分布のギブスサンプラー
        
        Parameters:
        -----------
        mu : array-like
            平均ベクトル
        Sigma : array-like  
            共分散行列
        """
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        
        # 条件付き分布のパラメータを事前計算
        self.sigma11 = Sigma[0, 0]
        self.sigma22 = Sigma[1, 1]
        self.sigma12 = Sigma[0, 1]
        
        # 条件付き分散
        self.cond_var1 = self.sigma11 - self.sigma12**2 / self.sigma22
        self.cond_var2 = self.sigma22 - self.sigma12**2 / self.sigma11
        
        self.samples = []
    
    def conditional_mean(self, var_idx, other_value):
        """条件付き平均の計算"""
        if var_idx == 0:  # X1 | X2
            return self.mu[0] + self.sigma12 / self.sigma22 * (other_value - self.mu[1])
        else:  # X2 | X1
            return self.mu[1] + self.sigma12 / self.sigma11 * (other_value - self.mu[0])
    
    def sample(self, n_samples, initial_value=None, burn_in=1000):
        """ギブスサンプリングの実行"""
        if initial_value is None:
            current_sample = np.array([0.0, 0.0])
        else:
            current_sample = np.array(initial_value)
        
        samples = []
        total_samples = burn_in + n_samples
        
        for i in range(total_samples):
            # X1 | X2 からサンプリング
            cond_mean1 = self.conditional_mean(0, current_sample[1])
            current_sample[0] = np.random.normal(cond_mean1, np.sqrt(self.cond_var1))
            
            # X2 | X1 からサンプリング
            cond_mean2 = self.conditional_mean(1, current_sample[0])
            current_sample[1] = np.random.normal(cond_mean2, np.sqrt(self.cond_var2))
            
            if i >= burn_in:
                samples.append(current_sample.copy())
        
        self.samples = np.array(samples)
        return self.samples
    
    def diagnostic_plots(self, true_samples=None):
        """診断プロットの作成"""
        if len(self.samples) == 0:
            print("サンプルが生成されていません。")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 散布図
        axes[0, 0].scatter(self.samples[:, 0], self.samples[:, 1], 
                          s=1, alpha=0.5, label='ギブスサンプル')
        if true_samples is not None:
            axes[0, 0].scatter(true_samples[:, 0], true_samples[:, 1], 
                              s=1, alpha=0.3, label='真のサンプル')
        axes[0, 0].set_xlabel('X1')
        axes[0, 0].set_ylabel('X2')
        axes[0, 0].set_title('サンプルの散布図')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # チェーンの軌跡（最初の100ステップ）
        trajectory = self.samples[:100]
        axes[0, 1].plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.7, linewidth=0.5)
        axes[0, 1].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='開始点')
        axes[0, 1].plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='終了点')
        axes[0, 1].set_xlabel('X1')
        axes[0, 1].set_ylabel('X2')
        axes[0, 1].set_title('ギブスサンプリングの軌跡（最初の100ステップ）')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # X1のトレースプロット
        axes[0, 2].plot(self.samples[:2000, 0])
        axes[0, 2].set_xlabel('イテレーション')
        axes[0, 2].set_ylabel('X1')
        axes[0, 2].set_title('X1のトレースプロット')
        axes[0, 2].grid(True, alpha=0.3)
        
        # X2のトレースプロット
        axes[1, 0].plot(self.samples[:2000, 1])
        axes[1, 0].set_xlabel('イテレーション')
        axes[1, 0].set_ylabel('X2')
        axes[1, 0].set_title('X2のトレースプロット')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 周辺分布の比較
        axes[1, 1].hist(self.samples[:, 0], bins=50, density=True, alpha=0.7, 
                       label='X1 (ギブス)')
        if true_samples is not None:
            axes[1, 1].hist(true_samples[:, 0], bins=50, density=True, alpha=0.5, 
                           label='X1 (真)')
        axes[1, 1].set_xlabel('X1')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].set_title('X1の周辺分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(self.samples[:, 1], bins=50, density=True, alpha=0.7, 
                       label='X2 (ギブス)')
        if true_samples is not None:
            axes[1, 2].hist(true_samples[:, 1], bins=50, density=True, alpha=0.5, 
                           label='X2 (真)')
        axes[1, 2].set_xlabel('X2')
        axes[1, 2].set_ylabel('密度')
        axes[1, 2].set_title('X2の周辺分布')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用例
def gibbs_sampling_demo():
    """ギブスサンプリングのデモンストレーション"""
    
    # パラメータ設定
    mu = [1.0, -0.5]
    Sigma = [[2.0, 1.5],
             [1.5, 2.0]]
    
    print("目標分布のパラメータ:")
    print(f"平均: {mu}")
    print(f"共分散行列:\n{np.array(Sigma)}")
    print(f"相関係数: {Sigma[0][1] / np.sqrt(Sigma[0][0] * Sigma[1][1]):.3f}")
    
    # ギブスサンプラー
    gibbs_sampler = GibbsSampler2D(mu, Sigma)
    gibbs_samples = gibbs_sampler.sample(n_samples=10000, initial_value=[0, 0])
    
    # 真のサンプル（比較用）
    true_samples = np.random.multivariate_normal(mu, Sigma, 10000)
    
    # 統計の比較
    print("\n統計量の比較:")
    print("             真の値    ギブス法")
    print(f"X1平均:      {np.mean(true_samples[:, 0]):7.3f}   {np.mean(gibbs_samples[:, 0]):7.3f}")
    print(f"X2平均:      {np.mean(true_samples[:, 1]):7.3f}   {np.mean(gibbs_samples[:, 1]):7.3f}")
    print(f"X1分散:      {np.var(true_samples[:, 0]):7.3f}   {np.var(gibbs_samples[:, 0]):7.3f}")
    print(f"X2分散:      {np.var(true_samples[:, 1]):7.3f}   {np.var(gibbs_samples[:, 1]):7.3f}")
    print(f"相関:        {np.corrcoef(true_samples.T)[0,1]:7.3f}   {np.corrcoef(gibbs_samples.T)[0,1]:7.3f}")
    
    # 診断プロット
    gibbs_sampler.diagnostic_plots(true_samples)

# 実行
gibbs_sampling_demo()
```

### 3.4 各手法の比較と使い分け

#### 3.4.1 性能比較実験

```python
def algorithm_comparison():
    """各MCMCアルゴリズムの性能比較"""
    
    # 目標分布：相関の強い2次元正規分布
    mu = [0, 0]
    rho = 0.8
    Sigma = [[1, rho],
             [rho, 1]]
    
    def target_logpdf(x):
        """目標分布の対数確率密度"""
        diff = x - np.array(mu)
        return -0.5 * diff @ np.linalg.inv(Sigma) @ diff
    
    # 1. メトロポリス法
    metro_sampler = MetropolisSampler(
        lambda x: target_logpdf(x) if np.isscalar(x) else np.array([target_logpdf(xi) for xi in x]),
        proposal_std=0.8
    )
    
    # 2. ギブスサンプリング
    gibbs_sampler = GibbsSampler2D(mu, Sigma)
    
    # サンプリング実行
    print("サンプリング実行中...")
    
    # メトロポリス法（1次元版のため、各次元を独立にサンプル）
    metro_samples_x1 = metro_sampler.sample(5000, initial_value=0.0)
    metro_sampler2 = MetropolisSampler(lambda x: target_logpdf([mu[0], x]), proposal_std=0.8)
    metro_samples_x2 = metro_sampler2.sample(5000, initial_value=0.0)
    metro_samples = np.column_stack([metro_samples_x1, metro_samples_x2])
    
    # ギブスサンプリング
    gibbs_samples = gibbs_sampler.sample(5000, initial_value=[0, 0])
    
    # 真のサンプル
    true_samples = np.random.multivariate_normal(mu, Sigma, 5000)
    
    # 結果の比較
    methods = ['真の分布', 'メトロポリス法', 'ギブスサンプリング']
    sample_sets = [true_samples, metro_samples, gibbs_samples]
    colors = ['blue', 'red', 'green']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (method, samples, color) in enumerate(zip(methods, sample_sets, colors)):
        # 散布図
        axes[0, i].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, color=color)
        axes[0, i].set_xlabel('X1')
        axes[0, i].set_ylabel('X2')
        axes[0, i].set_title(f'{method}\n平均: [{np.mean(samples[:, 0]):.3f}, {np.mean(samples[:, 1]):.3f}]')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')
        
        # 軌跡（最初の100ステップ）
        if i > 0:  # 真の分布以外
            trajectory = samples[:100]
            axes[1, i].plot(trajectory[:, 0], trajectory[:, 1], 
                           color=color, alpha=0.7, linewidth=0.5)
            axes[1, i].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=6)
            axes[1, i].plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=6)
            axes[1, i].set_xlabel('X1')
            axes[1, i].set_ylabel('X2')
            axes[1, i].set_title(f'{method}の軌跡')
            axes[1, i].grid(True, alpha=0.3)
        else:
            axes[1, i].hist2d(samples[:, 0], samples[:, 1], bins=30, alpha=0.7)
            axes[1, i].set_xlabel('X1')
            axes[1, i].set_ylabel('X2')
            axes[1, i].set_title('真の分布（2Dヒストグラム）')
    
    plt.tight_layout()
    plt.show()
    
    # 統計要約
    print("\n=== 性能比較結果 ===")
    for method, samples in zip(methods, sample_sets):
        corr = np.corrcoef(samples.T)[0, 1]
        print(f"\n{method}:")
        print(f"  相関係数: {corr:.3f} (真値: {rho:.3f})")
        print(f"  X1平均: {np.mean(samples[:, 0]):6.3f} (真値: {mu[0]:6.3f})")
        print(f"  X2平均: {np.mean(samples[:, 1]):6.3f} (真値: {mu[1]:6.3f})")
        print(f"  X1分散: {np.var(samples[:, 0]):6.3f} (真値: {Sigma[0][0]:6.3f})")
        print(f"  X2分散: {np.var(samples[:, 1]):6.3f} (真値: {Sigma[1][1]:6.3f})")

# 実行
algorithm_comparison()
```

#### 3.4.2 アルゴリズム選択指針

| アルゴリズム | 適用場面 | 長所 | 短所 |
|-------------|----------|------|------|
| **メトロポリス法** | 1次元問題、対称提案が自然 | 実装が簡単、直感的 | 高次元で効率低下 |
| **MH法** | 一般的な問題 | 最も汎用的、柔軟な提案設計 | 提案分布の調整が必要 |
| **ギブスサンプリング** | 条件付き分布が既知・サンプル可能 | 受容率100%、調整不要 | 適用範囲が限定的 |

### 3.5 実装のベストプラクティス

#### 3.5.1 数値安定性の確保

```python
def numerical_stability_tips():
    """数値安定性を確保するための実装テクニック"""
    
    def safe_log_acceptance(current_logp, proposal_logp, current_to_proposal_logq, proposal_to_current_logq):
        """数値安定性を考慮した受容確率の計算"""
        
        # 対数スケールで計算
        log_ratio = (proposal_logp + proposal_to_current_logq - 
                    current_logp - current_to_proposal_logq)
        
        # アンダーフロー/オーバーフローを避ける
        log_ratio = np.clip(log_ratio, -700, 700)  # exp(-700) ≈ 0, exp(700) ≈ inf
        
        # min(1, ratio) = min(0, log_ratio) in log scale
        log_acceptance = min(0.0, log_ratio)
        
        return np.exp(log_acceptance)
    
    def adaptive_proposal_scaling():
        """提案分布の適応的スケーリング"""
        
        class AdaptiveMetropolis:
            def __init__(self, target_acceptance=0.44, adaptation_rate=0.01):
                self.target_acceptance = target_acceptance
                self.adaptation_rate = adaptation_rate
                self.proposal_scale = 1.0
                self.n_accepted = 0
                self.n_total = 0
            
            def update_scale(self, accepted):
                """受容/棄却結果に基づいてスケールを更新"""
                self.n_total += 1
                if accepted:
                    self.n_accepted += 1
                
                # 定期的にスケールを調整
                if self.n_total % 100 == 0:
                    acceptance_rate = self.n_accepted / self.n_total
                    
                    if acceptance_rate > self.target_acceptance:
                        self.proposal_scale *= (1 + self.adaptation_rate)
                    else:
                        self.proposal_scale *= (1 - self.adaptation_rate)
                    
                    # スケールの範囲を制限
                    self.proposal_scale = np.clip(self.proposal_scale, 0.01, 10.0)
        
        return AdaptiveMetropolis()
    
    print("数値安定性のためのテクニック:")
    print("1. 対数スケールでの計算")
    print("2. クリッピングによるオーバーフロー防止")
    print("3. 適応的パラメータ調整")
    print("4. 極値の適切な処理")

# 実行
numerical_stability_tips()
```

この章では、MCMCの基本的なアルゴリズムを理論から実装まで詳しく解説しました。次章では、これらの手法を実際の問題に適用する実践的なワークフローを学びます。

---

## 第4章: 実践的応用とワークフロー

### 4.1 ベイズ線形回帰の完全実装

#### 4.1.1 問題設定と理論的背景

線形回帰は統計学の基本的な手法ですが、ベイズ的アプローチを取ることで、パラメータの不確実性を定量化し、予測区間を提供できます。

**モデル設定**：
$$y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

**事前分布**：
- $\boldsymbol{\beta} \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$
- $\sigma^2 \sim \text{InvGamma}(\alpha_0, \beta_0)$

**事後分布**：
$$p(\boldsymbol{\beta}, \sigma^2 | \mathbf{y}, \mathbf{X}) \propto p(\mathbf{y} | \mathbf{X}, \boldsymbol{\beta}, \sigma^2) p(\boldsymbol{\beta}) p(\sigma^2)$$

#### 4.1.2 ギブスサンプリングによる完全実装

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

class BayesianLinearRegression:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, mu_beta_prior=None, Sigma_beta_prior=None):
        """
        ベイズ線形回帰のギブスサンプラー
        
        Parameters:
        -----------
        alpha_prior, beta_prior : float
            σ²の事前分布 InvGamma(alpha_prior, beta_prior) のパラメータ
        mu_beta_prior : array-like or None
            βの事前分布の平均（Noneの場合は0ベクトル）
        Sigma_beta_prior : array-like or None  
            βの事前分布の共分散行列（Noneの場合は100*I）
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.mu_beta_prior = mu_beta_prior
        self.Sigma_beta_prior = Sigma_beta_prior
        
        # サンプル保存用
        self.beta_samples = []
        self.sigma2_samples = []
        self.fitted = False
    
    def fit(self, X, y, n_samples=10000, burn_in=2000, thin=1):
        """
        ギブスサンプリングによるベイズ推定
        
        Parameters:
        -----------
        X : array-like, shape (n, p)
            説明変数行列（切片項を含む）
        y : array-like, shape (n,)
            目的変数
        n_samples : int
            保存するサンプル数
        burn_in : int
            バーンイン期間
        thin : int
            間引き間隔
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        
        # 事前分布パラメータの設定
        if self.mu_beta_prior is None:
            self.mu_beta_prior = np.zeros(p)
        if self.Sigma_beta_prior is None:
            self.Sigma_beta_prior = 100.0 * np.eye(p)
        
        # 事前分布の精度行列
        Sigma_beta_prior_inv = np.linalg.inv(self.Sigma_beta_prior)
        
        # 初期値
        beta_current = np.random.normal(0, 1, p)
        sigma2_current = 1.0
        
        # サンプル保存用リスト
        beta_samples = []
        sigma2_samples = []
        
        total_iterations = burn_in + n_samples * thin
        
        print(f"ギブスサンプリング実行中... ({total_iterations} iterations)")
        
        for iteration in range(total_iterations):
            # Step 1: β | σ², y のサンプリング
            # 事後分布は多変量正規分布
            precision_matrix = X.T @ X / sigma2_current + Sigma_beta_prior_inv
            precision_matrix_inv = np.linalg.inv(precision_matrix)
            
            mean_vector = precision_matrix_inv @ (
                X.T @ y / sigma2_current + Sigma_beta_prior_inv @ self.mu_beta_prior
            )
            
            beta_current = np.random.multivariate_normal(mean_vector, precision_matrix_inv)
            
            # Step 2: σ² | β, y のサンプリング  
            # 事後分布は逆ガンマ分布
            residuals = y - X @ beta_current
            sse = np.sum(residuals**2)
            
            alpha_posterior = self.alpha_prior + n / 2
            beta_posterior = self.beta_prior + sse / 2
            
            sigma2_current = 1.0 / np.random.gamma(alpha_posterior, 1.0 / beta_posterior)
            
            # サンプルの保存（バーンイン後、間引きを考慮）
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                beta_samples.append(beta_current.copy())
                sigma2_samples.append(sigma2_current)
            
            # 進捗表示
            if (iteration + 1) % (total_iterations // 10) == 0:
                print(f"  進捗: {(iteration + 1) / total_iterations * 100:.1f}%")
        
        self.beta_samples = np.array(beta_samples)
        self.sigma2_samples = np.array(sigma2_samples)
        self.X_train = X
        self.y_train = y
        self.fitted = True
        
        print("サンプリング完了!")
        return self
    
    def predict(self, X_new, credible_interval=0.95):
        """
        予測分布からのサンプリングと予測区間の計算
        
        Parameters:
        -----------
        X_new : array-like, shape (n_new, p)
            予測したい説明変数
        credible_interval : float
            信頼区間の水準
            
        Returns:
        --------
        predictions : dict
            平均、中央値、信頼区間を含む予測結果
        """
        if not self.fitted:
            raise ValueError("モデルが学習されていません。fitメソッドを先に実行してください。")
        
        X_new = np.asarray(X_new)
        n_new = X_new.shape[0]
        n_samples = len(self.beta_samples)
        
        # 予測分布からのサンプリング
        y_pred_samples = np.zeros((n_samples, n_new))
        
        for i in range(n_samples):
            # 平均の予測
            mu_pred = X_new @ self.beta_samples[i]
            # 予測分布からサンプリング（観測ノイズを含む）
            y_pred_samples[i] = np.random.normal(mu_pred, np.sqrt(self.sigma2_samples[i]))
        
        # 統計量の計算
        alpha = 1 - credible_interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        predictions = {
            'mean': np.mean(y_pred_samples, axis=0),
            'median': np.median(y_pred_samples, axis=0),
            'std': np.std(y_pred_samples, axis=0),
            'lower_ci': np.percentile(y_pred_samples, lower_percentile, axis=0),
            'upper_ci': np.percentile(y_pred_samples, upper_percentile, axis=0),
            'samples': y_pred_samples
        }
        
        return predictions
    
    def summary(self):
        """推定結果の要約"""
        if not self.fitted:
            raise ValueError("モデルが学習されていません。")
        
        print("=== ベイズ線形回帰 推定結果 ===")
        print(f"サンプル数: {len(self.beta_samples)}")
        print(f"パラメータ数: {self.beta_samples.shape[1]}")
        
        print("\n回帰係数の事後統計量:")
        for i in range(self.beta_samples.shape[1]):
            beta_i = self.beta_samples[:, i]
            print(f"  β_{i}: 平均={np.mean(beta_i):7.4f}, "
                  f"中央値={np.median(beta_i):7.4f}, "
                  f"標準偏差={np.std(beta_i):7.4f}")
            
            # 95%信頼区間
            ci_lower = np.percentile(beta_i, 2.5)
            ci_upper = np.percentile(beta_i, 97.5)
            print(f"       95%CI: [{ci_lower:7.4f}, {ci_upper:7.4f}]")
        
        print(f"\n誤差分散σ²:")
        print(f"  平均: {np.mean(self.sigma2_samples):7.4f}")
        print(f"  中央値: {np.median(self.sigma2_samples):7.4f}")
        print(f"  標準偏差: {np.std(self.sigma2_samples):7.4f}")
        
        # 95%信頼区間
        ci_lower = np.percentile(self.sigma2_samples, 2.5)
        ci_upper = np.percentile(self.sigma2_samples, 97.5)
        print(f"  95%CI: [{ci_lower:7.4f}, {ci_upper:7.4f}]")
    
    def plot_diagnostics(self):
        """診断プロットの作成"""
        if not self.fitted:
            raise ValueError("モデルが学習されていません。")
        
        n_params = self.beta_samples.shape[1]
        fig, axes = plt.subplots(n_params + 1, 3, figsize=(15, 4 * (n_params + 1)))
        
        # 回帰係数のプロット
        for i in range(n_params):
            beta_i = self.beta_samples[:, i]
            
            # トレースプロット
            axes[i, 0].plot(beta_i)
            axes[i, 0].set_title(f'β_{i} トレースプロット')
            axes[i, 0].set_ylabel(f'β_{i}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 事後分布
            axes[i, 1].hist(beta_i, bins=50, density=True, alpha=0.7)
            axes[i, 1].set_title(f'β_{i} 事後分布')
            axes[i, 1].set_xlabel(f'β_{i}')
            axes[i, 1].set_ylabel('密度')
            axes[i, 1].grid(True, alpha=0.3)
            
            # 自己相関
            def autocorr(x, max_lag=50):
                x = x - np.mean(x)
                autocorrs = np.correlate(x, x, mode='full')
                autocorrs = autocorrs[autocorrs.size // 2:]
                autocorrs = autocorrs / autocorrs[0]
                return autocorrs[:max_lag]
            
            lags = range(50)
            autocorrs = autocorr(beta_i)
            axes[i, 2].plot(lags, autocorrs)
            axes[i, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
            axes[i, 2].set_title(f'β_{i} 自己相関')
            axes[i, 2].set_xlabel('ラグ')
            axes[i, 2].set_ylabel('自己相関')
            axes[i, 2].grid(True, alpha=0.3)
        
        # σ²のプロット
        # トレースプロット
        axes[n_params, 0].plot(self.sigma2_samples)
        axes[n_params, 0].set_title('σ² トレースプロット')
        axes[n_params, 0].set_ylabel('σ²')
        axes[n_params, 0].set_xlabel('イテレーション')
        axes[n_params, 0].grid(True, alpha=0.3)
        
        # 事後分布
        axes[n_params, 1].hist(self.sigma2_samples, bins=50, density=True, alpha=0.7)
        axes[n_params, 1].set_title('σ² 事後分布')
        axes[n_params, 1].set_xlabel('σ²')
        axes[n_params, 1].set_ylabel('密度')
        axes[n_params, 1].grid(True, alpha=0.3)
        
        # 自己相関
        autocorrs = autocorr(self.sigma2_samples)
        axes[n_params, 2].plot(lags, autocorrs)
        axes[n_params, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[n_params, 2].set_title('σ² 自己相関')
        axes[n_params, 2].set_xlabel('ラグ')
        axes[n_params, 2].set_ylabel('自己相関')
        axes[n_params, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用例：合成データでのデモンストレーション
def bayesian_regression_demo():
    """ベイズ線形回帰のデモンストレーション"""
    
    # 合成データの生成
    np.random.seed(42)
    n = 100
    true_beta = np.array([2.0, -1.5, 0.8])  # [切片, x1の係数, x2の係数]
    true_sigma = 0.5
    
    # 説明変数
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)  
    X = np.column_stack([np.ones(n), x1, x2])  # 切片項を追加
    
    # 目的変数
    y = X @ true_beta + np.random.normal(0, true_sigma, n)
    
    print("=== 合成データ ===")
    print(f"サンプルサイズ: {n}")
    print(f"真のパラメータ: β = {true_beta}")
    print(f"真の誤差標準偏差: σ = {true_sigma}")
    
    # ベイズ線形回帰の実行
    model = BayesianLinearRegression(
        alpha_prior=1.0,  # σ²の事前分布パラメータ
        beta_prior=1.0,
        mu_beta_prior=np.zeros(3),  # βの事前平均
        Sigma_beta_prior=10.0 * np.eye(3)  # βの事前共分散
    )
    
    model.fit(X, y, n_samples=8000, burn_in=2000)
    
    # 結果の要約
    model.summary()
    
    # 診断プロット
    model.plot_diagnostics()
    
    # 予測の実行
    print("\n=== 予測 ===")
    X_test = np.array([[1, 0.5, -0.3],   # 切片, x1, x2
                       [1, -1.0, 1.2]])
    
    predictions = model.predict(X_test, credible_interval=0.95)
    
    for i, (x1_val, x2_val) in enumerate(X_test[:, 1:]):
        print(f"テストポイント {i+1}: x1={x1_val}, x2={x2_val}")
        print(f"  予測平均: {predictions['mean'][i]:.3f}")
        print(f"  95%予測区間: [{predictions['lower_ci'][i]:.3f}, {predictions['upper_ci'][i]:.3f}]")
        
        # 真の値と比較
        true_y = X_test[i] @ true_beta
        print(f"  真の値（ノイズなし）: {true_y:.3f}")
    
    return model, X, y, predictions

# デモ実行
model, X, y, predictions = bayesian_regression_demo()
```

#### 4.1.3 予測性能の評価と可視化

```python
def prediction_visualization(model, X, y):
    """予測結果の可視化"""
    
    # 学習データに対する予測
    train_predictions = model.predict(X)
    
    # 残差分析
    residuals = y - train_predictions['mean']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 実測値 vs 予測値
    axes[0, 0].scatter(y, train_predictions['mean'], alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('実測値')
    axes[0, 0].set_ylabel('予測値')
    axes[0, 0].set_title('実測値 vs 予測値')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差プロット
    axes[0, 1].scatter(train_predictions['mean'], residuals, alpha=0.6)
    axes[0, 1].axhline(0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('予測値')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差プロット')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差のヒストグラム
    axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('残差')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('残差の分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Qプロット（正規性の確認）
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('残差のQ-Qプロット')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 予測区間の可視化（1次元の場合）
    if X.shape[1] == 2:  # 切片 + 1変数の場合
        x_plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        X_plot = np.column_stack([np.ones(100), x_plot])
        
        plot_predictions = model.predict(X_plot)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 1], y, alpha=0.6, label='観測データ')
        plt.plot(x_plot, plot_predictions['mean'], 'r-', linewidth=2, label='予測平均')
        plt.fill_between(x_plot, 
                        plot_predictions['lower_ci'], 
                        plot_predictions['upper_ci'],
                        alpha=0.3, color='red', label='95%予測区間')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('ベイズ線形回帰の予測区間')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# 予測可視化の実行
prediction_visualization(model, X, y)
```

### 4.2 階層ベイズモデルの構築

階層ベイズモデルは、データに自然な階層構造がある場合に威力を発揮します。例えば、複数の学校の学生の成績、複数の病院の治療効果など。

#### 4.2.1 階層ベイズモデルの理論

**問題設定**：$J$個のグループがあり、各グループ$j$について$n_j$個の観測値$y_{ij}$がある。

**モデル**：
- レベル1: $y_{ij} | \theta_j, \sigma^2 \sim \mathcal{N}(\theta_j, \sigma^2)$
- レベル2: $\theta_j | \mu, \tau^2 \sim \mathcal{N}(\mu, \tau^2)$
- レベル3: $\mu \sim \mathcal{N}(\mu_0, \sigma_\mu^2)$, $\tau^2 \sim \text{InvGamma}(\alpha_\tau, \beta_\tau)$, $\sigma^2 \sim \text{InvGamma}(\alpha_\sigma, \beta_\sigma)$

#### 4.2.2 実装例：学校の効果分析

```python
class HierarchicalBayesianModel:
    def __init__(self, mu_prior_mean=0, mu_prior_var=100, 
                 tau2_prior_alpha=1, tau2_prior_beta=1,
                 sigma2_prior_alpha=1, sigma2_prior_beta=1):
        """
        階層ベイズモデル
        
        Parameters:
        -----------
        mu_prior_mean, mu_prior_var : float
            μの事前分布 N(mu_prior_mean, mu_prior_var)
        tau2_prior_alpha, tau2_prior_beta : float
            τ²の事前分布 InvGamma(alpha, beta)
        sigma2_prior_alpha, sigma2_prior_beta : float
            σ²の事前分布 InvGamma(alpha, beta)
        """
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_var = mu_prior_var
        self.tau2_prior_alpha = tau2_prior_alpha
        self.tau2_prior_beta = tau2_prior_beta
        self.sigma2_prior_alpha = sigma2_prior_alpha
        self.sigma2_prior_beta = sigma2_prior_beta
        
        self.fitted = False
    
    def fit(self, group_data, n_samples=10000, burn_in=2000):
        """
        階層ベイズモデルの推定
        
        Parameters:
        -----------
        group_data : list of arrays
            各グループのデータ
        n_samples : int
            保存するサンプル数
        burn_in : int
            バーンイン期間
        """
        self.group_data = group_data
        self.n_groups = len(group_data)
        self.group_sizes = [len(data) for data in group_data]
        self.group_means = [np.mean(data) for data in group_data]
        
        print(f"階層ベイズモデル推定開始...")
        print(f"グループ数: {self.n_groups}")
        print(f"各グループのサンプルサイズ: {self.group_sizes}")
        
        # 初期値
        theta_current = np.array(self.group_means)  # 各グループの平均
        mu_current = np.mean(self.group_means)      # 全体平均
        tau2_current = 1.0                          # グループ間分散
        sigma2_current = 1.0                        # グループ内分散
        
        # サンプル保存用
        theta_samples = []
        mu_samples = []
        tau2_samples = []
        sigma2_samples = []
        
        total_iterations = burn_in + n_samples
        
        for iteration in range(total_iterations):
            # Step 1: θ_j | μ, τ², σ², y のサンプリング
            for j in range(self.n_groups):
                n_j = self.group_sizes[j]
                y_bar_j = self.group_means[j]
                
                # 事後平均と分散の計算
                precision = n_j / sigma2_current + 1 / tau2_current
                posterior_var = 1 / precision
                posterior_mean = posterior_var * (n_j * y_bar_j / sigma2_current + mu_current / tau2_current)
                
                theta_current[j] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
            
            # Step 2: μ | θ, τ² のサンプリング
            precision_mu = self.n_groups / tau2_current + 1 / self.mu_prior_var
            posterior_var_mu = 1 / precision_mu
            posterior_mean_mu = posterior_var_mu * (
                np.sum(theta_current) / tau2_current + self.mu_prior_mean / self.mu_prior_var
            )
            
            mu_current = np.random.normal(posterior_mean_mu, np.sqrt(posterior_var_mu))
            
            # Step 3: τ² | μ, θ のサンプリング
            alpha_post = self.tau2_prior_alpha + self.n_groups / 2
            beta_post = self.tau2_prior_beta + np.sum((theta_current - mu_current)**2) / 2
            
            tau2_current = 1 / np.random.gamma(alpha_post, 1 / beta_post)
            
            # Step 4: σ² | θ, y のサンプリング
            total_n = sum(self.group_sizes)
            sse = 0
            for j in range(self.n_groups):
                sse += np.sum((self.group_data[j] - theta_current[j])**2)
            
            alpha_post = self.sigma2_prior_alpha + total_n / 2
            beta_post = self.sigma2_prior_beta + sse / 2
            
            sigma2_current = 1 / np.random.gamma(alpha_post, 1 / beta_post)
            
            # サンプルの保存
            if iteration >= burn_in:
                theta_samples.append(theta_current.copy())
                mu_samples.append(mu_current)
                tau2_samples.append(tau2_current)
                sigma2_samples.append(sigma2_current)
            
            # 進捗表示
            if (iteration + 1) % (total_iterations // 10) == 0:
                print(f"  進捗: {(iteration + 1) / total_iterations * 100:.1f}%")
        
        self.theta_samples = np.array(theta_samples)
        self.mu_samples = np.array(mu_samples)
        self.tau2_samples = np.array(tau2_samples)
        self.sigma2_samples = np.array(sigma2_samples)
        self.fitted = True
        
        print("推定完了!")
        return self
    
    def summary(self):
        """推定結果の要約"""
        if not self.fitted:
            raise ValueError("モデルが学習されていません。")
        
        print("=== 階層ベイズモデル 推定結果 ===")
        
        # 全体平均μ
        print(f"\n全体平均 μ:")
        print(f"  事後平均: {np.mean(self.mu_samples):.4f}")
        print(f"  事後標準偏差: {np.std(self.mu_samples):.4f}")
        print(f"  95%信頼区間: [{np.percentile(self.mu_samples, 2.5):.4f}, "
              f"{np.percentile(self.mu_samples, 97.5):.4f}]")
        
        # グループ間分散τ²
        print(f"\nグループ間分散 τ²:")
        print(f"  事後平均: {np.mean(self.tau2_samples):.4f}")
        print(f"  事後標準偏差: {np.std(self.tau2_samples):.4f}")
        print(f"  95%信頼区間: [{np.percentile(self.tau2_samples, 2.5):.4f}, "
              f"{np.percentile(self.tau2_samples, 97.5):.4f}]")
        
        # グループ内分散σ²
        print(f"\nグループ内分散 σ²:")
        print(f"  事後平均: {np.mean(self.sigma2_samples):.4f}")
        print(f"  事後標準偏差: {np.std(self.sigma2_samples):.4f}")
        print(f"  95%信頼区間: [{np.percentile(self.sigma2_samples, 2.5):.4f}, "
              f"{np.percentile(self.sigma2_samples, 97.5):.4f}]")
        
        # 各グループの効果θ_j
        print(f"\n各グループの効果 θ_j:")
        for j in range(self.n_groups):
            theta_j = self.theta_samples[:, j]
            print(f"  グループ {j+1}: 事後平均={np.mean(theta_j):.4f}, "
                  f"標準偏差={np.std(theta_j):.4f}, "
                  f"95%CI=[{np.percentile(theta_j, 2.5):.4f}, {np.percentile(theta_j, 97.5):.4f}]")
    
    def plot_results(self):
        """結果の可視化"""
        if not self.fitted:
            raise ValueError("モデルが学習されていません。")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # グループ効果の比較
        theta_means = np.mean(self.theta_samples, axis=0)
        theta_ci_lower = np.percentile(self.theta_samples, 2.5, axis=0)
        theta_ci_upper = np.percentile(self.theta_samples, 97.5, axis=0)
        
        x_pos = np.arange(self.n_groups)
        axes[0, 0].errorbar(x_pos, theta_means, 
                           yerr=[theta_means - theta_ci_lower, theta_ci_upper - theta_means],
                           fmt='o', capsize=5)
        
        # 観測データの平均も追加
        axes[0, 0].scatter(x_pos, self.group_means, color='red', s=100, alpha=0.7, 
                          label='観測平均')
        
        axes[0, 0].set_xlabel('グループ')
        axes[0, 0].set_ylabel('効果')
        axes[0, 0].set_title('各グループの効果 (95%信頼区間)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 全体平均μのトレース
        axes[0, 1].plot(self.mu_samples)
        axes[0, 1].set_title('全体平均 μ のトレース')
        axes[0, 1].set_ylabel('μ')
        axes[0, 1].grid(True, alpha=0.3)
        
        # グループ間分散τ²のトレース
        axes[0, 2].plot(self.tau2_samples)
        axes[0, 2].set_title('グループ間分散 τ² のトレース')
        axes[0, 2].set_ylabel('τ²')
        axes[0, 2].grid(True, alpha=0.3)
        
        # グループ内分散σ²のトレース
        axes[1, 0].plot(self.sigma2_samples)
        axes[1, 0].set_title('グループ内分散 σ² のトレース')
        axes[1, 0].set_ylabel('σ²')
        axes[1, 0].set_xlabel('イテレーション')
        axes[1, 0].grid(True, alpha=0.3)
        
        # パラメータの事後分布
        axes[1, 1].hist(self.mu_samples, bins=50, alpha=0.7, label='μ')
        axes[1, 1].set_title('全体平均 μ の事後分布')
        axes[1, 1].set_xlabel('μ')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 分散成分の比較
        variance_ratio = self.tau2_samples / (self.tau2_samples + self.sigma2_samples)
        axes[1, 2].hist(variance_ratio, bins=50, alpha=0.7)
        axes[1, 2].set_title('グループ間分散の割合 τ²/(τ²+σ²)')
        axes[1, 2].set_xlabel('分散比')
        axes[1, 2].set_ylabel('密度')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用例：8つの学校の効果分析（有名なデータセット）
def hierarchical_model_demo():
    """階層ベイズモデルのデモンストレーション"""
    
    # 8つの学校の観測効果（標準誤差も考慮した簡単な例）
    school_effects = [28, 8, -3, 7, -1, 1, 18, 12]
    school_ses = [15, 10, 16, 11, 9, 11, 10, 18]
    
    # 観測値を正規分布からシミュレート
    np.random.seed(123)
    school_data = []
    for i, (effect, se) in enumerate(zip(school_effects, school_ses)):
        # 各学校について複数の観測値を生成
        n_obs = 20  # 各学校20人の学生
        data = np.random.normal(effect, se, n_obs)
        school_data.append(data)
    
    print("=== 8つの学校の効果分析 ===")
    print("各学校の観測統計量:")
    for i, data in enumerate(school_data):
        print(f"学校 {i+1}: 平均={np.mean(data):6.2f}, 標準偏差={np.std(data):6.2f}, n={len(data)}")
    
    # 階層ベイズモデルの推定
    model = HierarchicalBayesianModel(
        mu_prior_mean=0,      # 全体平均の事前平均
        mu_prior_var=100,     # 全体平均の事前分散
        tau2_prior_alpha=1,   # グループ間分散の事前分布
        tau2_prior_beta=1,
        sigma2_prior_alpha=1, # グループ内分散の事前分布
        sigma2_prior_beta=1
    )
    
    model.fit(school_data, n_samples=8000, burn_in=2000)
    
    # 結果の要約と可視化
    model.summary()
    model.plot_results()
    
    return model, school_data

# デモ実行
hierarchical_model, school_data = hierarchical_model_demo()
```

### 4.3 機械学習への応用：潜在変数モデル

#### 4.3.1 ベイズ混合ガウスモデル

```python
class BayesianGaussianMixture:
    def __init__(self, n_components, alpha_prior=1.0, beta_prior=1.0, 
                 mu_prior_mean=0.0, mu_prior_precision=1.0):
        """
        ベイズ混合ガウスモデル
        
        Parameters:
        -----------
        n_components : int
            混合成分数
        alpha_prior : float
            混合比率の事前分布パラメータ（ディリクレ分布）
        beta_prior : float
            精度の事前分布パラメータ（ガンマ分布）
        mu_prior_mean : float
            平均の事前分布の平均
        mu_prior_precision : float
            平均の事前分布の精度
        """
        self.n_components = n_components
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_precision = mu_prior_precision
        
        self.fitted = False
    
    def fit(self, X, n_samples=5000, burn_in=1000):
        """
        ギブスサンプリングによるベイズ推定
        
        Parameters:
        -----------
        X : array-like, shape (n, 1)
            観測データ（1次元）
        n_samples : int
            保存するサンプル数
        burn_in : int
            バーンイン期間
        """
        X = np.asarray(X).flatten()
        n = len(X)
        K = self.n_components
        
        print(f"ベイズ混合ガウスモデル推定開始...")
        print(f"データ数: {n}, 混合成分数: {K}")
        
        # 初期値
        # 潜在変数（各データポイントがどの成分に属するか）
        z = np.random.randint(0, K, n)
        
        # パラメータの初期値
        pi = np.ones(K) / K  # 混合比率
        mu = np.random.normal(np.mean(X), np.std(X), K)  # 各成分の平均
        tau = np.ones(K)  # 各成分の精度（分散の逆数）
        
        # サンプル保存用
        z_samples = []
        pi_samples = []
        mu_samples = []
        tau_samples = []
        
        total_iterations = burn_in + n_samples
        
        for iteration in range(total_iterations):
            # Step 1: 潜在変数 z のサンプリング
            for i in range(n):
                # 各成分に属する確率を計算
                log_probs = np.zeros(K)
                for k in range(K):
                    # log p(x_i | mu_k, tau_k) + log pi_k
                    log_probs[k] = (np.log(pi[k]) + 
                                   0.5 * np.log(tau[k]) - 
                                   0.5 * tau[k] * (X[i] - mu[k])**2)
                
                # 正規化してソフトマックス
                max_log_prob = np.max(log_probs)
                log_probs -= max_log_prob
                probs = np.exp(log_probs)
                probs /= np.sum(probs)
                
                # サンプリング
                z[i] = np.random.choice(K, p=probs)
            
            # Step 2: 混合比率 π のサンプリング（ディリクレ分布）
            counts = np.bincount(z, minlength=K)
            alpha_post = self.alpha_prior + counts
            pi = np.random.dirichlet(alpha_post)
            
            # Step 3: 各成分の平均 μ_k のサンプリング
            for k in range(K):
                # k番目の成分に属するデータ
                X_k = X[z == k]
                n_k = len(X_k)
                
                if n_k > 0:
                    # 事後分布の計算
                    precision_post = self.mu_prior_precision + n_k * tau[k]
                    mean_post = ((self.mu_prior_precision * self.mu_prior_mean + 
                                tau[k] * np.sum(X_k)) / precision_post)
                    
                    mu[k] = np.random.normal(mean_post, 1/np.sqrt(precision_post))
                else:
                    # データが割り当てられていない場合は事前分布からサンプル
                    mu[k] = np.random.normal(self.mu_prior_mean, 
                                           1/np.sqrt(self.mu_prior_precision))
            
            # Step 4: 各成分の精度 τ_k のサンプリング（ガンマ分布）
            for k in range(K):
                X_k = X[z == k]
                n_k = len(X_k)
                
                if n_k > 0:
                    # 事後分布のパラメータ
                    alpha_post = self.beta_prior + n_k / 2
                    beta_post = self.beta_prior + 0.5 * np.sum((X_k - mu[k])**2)
                    
                    tau[k] = np.random.gamma(alpha_post, 1/beta_post)
                else:
                    # データが割り当てられていない場合は事前分布からサンプル
                    tau[k] = np.random.gamma(self.beta_prior, 1/self.beta_prior)
            
            # サンプルの保存
            if iteration >= burn_in:
                z_samples.append(z.copy())
                pi_samples.append(pi.copy())
                mu_samples.append(mu.copy())
                tau_samples.append(tau.copy())
            
            # 進捗表示
            if (iteration + 1) % (total_iterations // 10) == 0:
                print(f"  進捗: {(iteration + 1) / total_iterations * 100:.1f}%")
        
        self.X = X
        self.z_samples = np.array(z_samples)
        self.pi_samples = np.array(pi_samples)
        self.mu_samples = np.array(mu_samples)
        self.tau_samples = np.array(tau_samples)
        self.sigma_samples = 1.0 / np.sqrt(self.tau_samples)  # 標準偏差
        self.fitted = True
        
        print("推定完了!")
        return self
    
    def plot_results(self):
        """結果の可視化"""
        if not self.fitted:
            raise ValueError("モデルが学習されていません。")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # データのヒストグラムと推定された混合分布
        axes[0, 0].hist(self.X, bins=30, density=True, alpha=0.7, color='skyblue')
        
        # 事後平均を使った混合分布の描画
        x_range = np.linspace(self.X.min(), self.X.max(), 200)
        mu_mean = np.mean(self.mu_samples, axis=0)
        sigma_mean = np.mean(self.sigma_samples, axis=0)
        pi_mean = np.mean(self.pi_samples, axis=0)
        
        total_density = np.zeros_like(x_range)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for k in range(self.n_components):
            component_density = (pi_mean[k] * 
                               stats.norm.pdf(x_range, mu_mean[k], sigma_mean[k]))
            total_density += component_density
            
            axes[0, 0].plot(x_range, component_density, 
                           color=colors[k % len(colors)], linestyle='--',
                           label=f'成分 {k+1}')
        
        axes[0, 0].plot(x_range, total_density, 'k-', linewidth=2, label='混合分布')
        axes[0, 0].set_title('データと推定された混合分布')
        axes[0, 0].set_xlabel('値')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 混合比率のトレース
        for k in range(self.n_components):
            axes[0, 1].plot(self.pi_samples[:, k], 
                           color=colors[k % len(colors)], label=f'π_{k+1}')
        axes[0, 1].set_title('混合比率のトレース')
        axes[0, 1].set_ylabel('π')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 平均のトレース
        for k in range(self.n_components):
            axes[0, 2].plot(self.mu_samples[:, k], 
                           color=colors[k % len(colors)], label=f'μ_{k+1}')
        axes[0, 2].set_title('平均のトレース')
        axes[0, 2].set_ylabel('μ')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 標準偏差のトレース
        for k in range(self.n_components):
            axes[1, 0].plot(self.sigma_samples[:, k], 
                           color=colors[k % len(colors)], label=f'σ_{k+1}')
        axes[1, 0].set_title('標準偏差のトレース')
        axes[1, 0].set_xlabel('イテレーション')
        axes[1, 0].set_ylabel('σ')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # パラメータの事後分布
        for k in range(self.n_components):
            axes[1, 1].hist(self.mu_samples[:, k], bins=50, alpha=0.6,
                           color=colors[k % len(colors)], label=f'μ_{k+1}')
        axes[1, 1].set_title('平均の事後分布')
        axes[1, 1].set_xlabel('μ')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 混合比率の事後分布
        for k in range(self.n_components):
            axes[1, 2].hist(self.pi_samples[:, k], bins=50, alpha=0.6,
                           color=colors[k % len(colors)], label=f'π_{k+1}')
        axes[1, 2].set_title('混合比率の事後分布')
        axes[1, 2].set_xlabel('π')
        axes[1, 2].set_ylabel('密度')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用例：混合ガウス分布からのデータ
def mixture_model_demo():
    """ベイズ混合ガウスモデルのデモンストレーション"""
    
    # 真の混合分布からデータを生成
    np.random.seed(42)
    n = 300
    
    # 真のパラメータ
    true_pi = [0.3, 0.5, 0.2]
    true_mu = [-2, 1, 4]
    true_sigma = [0.8, 1.2, 0.6]
    
    # データ生成
    X = []
    true_labels = []
    
    for i in range(n):
        # 成分を選択
        component = np.random.choice(3, p=true_pi)
        # データポイントを生成
        x = np.random.normal(true_mu[component], true_sigma[component])
        X.append(x)
        true_labels.append(component)
    
    X = np.array(X)
    true_labels = np.array(true_labels)
    
    print("=== 混合ガウス分布からのデータ ===")
    print(f"データ数: {n}")
    print(f"真のパラメータ:")
    for k in range(3):
        print(f"  成分 {k+1}: π={true_pi[k]:.2f}, μ={true_mu[k]:.2f}, σ={true_sigma[k]:.2f}")
    
    # モデルの推定
    model = BayesianGaussianMixture(
        n_components=3,
        alpha_prior=1.0,        # 混合比率の事前分布
        beta_prior=1.0,         # 精度の事前分布
        mu_prior_mean=0.0,      # 平均の事前分布
        mu_prior_precision=0.1  # 平均の事前精度
    )
    
    model.fit(X, n_samples=6000, burn_in=2000)
    
    # 結果の可視化
    model.plot_results()
    
    # 推定結果の要約
    print("\n=== 推定結果 ===")
    pi_mean = np.mean(model.pi_samples, axis=0)
    mu_mean = np.mean(model.mu_samples, axis=0)
    sigma_mean = np.mean(model.sigma_samples, axis=0)
    
    # 成分を真の値と対応させる（ラベルスイッチング問題を簡単に処理）
    # 平均でソート
    sorted_indices = np.argsort(mu_mean)
    
    print("推定されたパラメータ（平均でソート後）:")
    for i, k in enumerate(sorted_indices):
        print(f"  成分 {i+1}: π={pi_mean[k]:.3f}, μ={mu_mean[k]:.3f}, σ={sigma_mean[k]:.3f}")
        if i < len(true_pi):
            print(f"          (真値: π={true_pi[i]:.3f}, μ={true_mu[i]:.3f}, σ={true_sigma[i]:.3f})")
    
    return model, X, true_labels

# デモ実行
mixture_model, X_mixture, true_labels = mixture_model_demo()
```

### 4.4 統計物理学への応用：イジングモデル

#### 4.4.1 2次元イジングモデルのシミュレーション

```python
class IsingModel2D:
    def __init__(self, size, temperature=2.3, external_field=0.0):
        """
        2次元イジングモデル
        
        Parameters:
        -----------
        size : int
            格子のサイズ (size x size)
        temperature : float
            温度 (臨界温度は約2.269)
        external_field : float
            外部磁場
        """
        self.size = size
        self.temperature = temperature
        self.external_field = external_field
        self.beta = 1.0 / temperature if temperature > 0 else float('inf')
        
        # スピン配置の初期化（+1 または -1）
        self.spins = np.random.choice([-1, 1], size=(size, size))
        
        # 統計量保存用
        self.energy_history = []
        self.magnetization_history = []
        self.acceptance_history = []
    
    def local_energy(self, i, j):
        """
        位置(i,j)のスピンの局所エネルギーを計算
        """
        # 周期境界条件
        up = (i - 1) % self.size
        down = (i + 1) % self.size
        left = (j - 1) % self.size
        right = (j + 1) % self.size
        
        # 近接スピンとの相互作用
        neighbors_sum = (self.spins[up, j] + self.spins[down, j] + 
                        self.spins[i, left] + self.spins[i, right])
        
        # エネルギー = -J * スピン * 近接スピンの和 - h * スピン
        # ここではJ=1とする
        energy = -self.spins[i, j] * neighbors_sum - self.external_field * self.spins[i, j]
        
        return energy
    
    def total_energy(self):
        """全エネルギーの計算"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                energy += self.local_energy(i, j)
        
        # 相互作用エネルギーの二重カウントを修正
        return energy / 2
    
    def magnetization(self):
        """磁化の計算"""
        return np.sum(self.spins)
    
    def metropolis_step(self):
        """メトロポリス法による1ステップの更新"""
        n_accepted = 0
        
        # 全格子点について更新を試行
        for _ in range(self.size * self.size):
            # ランダムな格子点を選択
            i = np.random.randint(0, self.size)
            j = np.random.randint(0, self.size)
            
            # 現在のエネルギー
            current_energy = self.local_energy(i, j)
            
            # スピンを反転
            self.spins[i, j] *= -1
            
            # 新しいエネルギー
            new_energy = self.local_energy(i, j)
            
            # エネルギー変化
            delta_E = new_energy - current_energy
            
            # 受容確率の計算
            if delta_E <= 0:
                # エネルギーが下がる場合は必ず受容
                n_accepted += 1
            else:
                # エネルギーが上がる場合はボルツマン因子で判定
                prob = np.exp(-self.beta * delta_E)
                if np.random.random() < prob:
                    n_accepted += 1
                else:
                    # 棄却：スピンを元に戻す
                    self.spins[i, j] *= -1
        
        return n_accepted / (self.size * self.size)
    
    def simulate(self, n_steps, record_interval=10):
        """
        モンテカルロシミュレーションの実行
        
        Parameters:
        -----------
        n_steps : int
            シミュレーションステップ数
        record_interval : int
            統計量を記録する間隔
        """
        print(f"イジングモデルシミュレーション開始...")
        print(f"格子サイズ: {self.size}x{self.size}")
        print(f"温度: {self.temperature:.3f}")
        print(f"外部磁場: {self.external_field:.3f}")
        
        self.energy_history = []
        self.magnetization_history = []
        self.acceptance_history = []
        
        for step in range(n_steps):
            # メトロポリス更新
            acceptance_rate = self.metropolis_step()
            
            # 統計量の記録
            if step % record_interval == 0:
                energy = self.total_energy()
                magnetization = self.magnetization()
                
                self.energy_history.append(energy)
                self.magnetization_history.append(magnetization)
                self.acceptance_history.append(acceptance_rate)
            
            # 進捗表示
            if (step + 1) % (n_steps // 10) == 0:
                print(f"  進捗: {(step + 1) / n_steps * 100:.1f}%")
        
        print("シミュレーション完了!")
        return self
    
    def plot_results(self):
        """結果の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # スピン配置の可視化
        im = axes[0, 0].imshow(self.spins, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, 0].set_title(f'スピン配置 (T={self.temperature:.3f})')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im, ax=axes[0, 0])
        
        # エネルギーの時系列
        steps = np.arange(len(self.energy_history)) * 10
        axes[0, 1].plot(steps, self.energy_history)
        axes[0, 1].set_title('エネルギーの時間発展')
        axes[0, 1].set_xlabel('MCステップ')
        axes[0, 1].set_ylabel('エネルギー')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 磁化の時系列
        axes[0, 2].plot(steps, self.magnetization_history)
        axes[0, 2].set_title('磁化の時間発展')
        axes[0, 2].set_xlabel('MCステップ')
        axes[0, 2].set_ylabel('磁化')
        axes[0, 2].grid(True, alpha=0.3)
        
        # エネルギーの分布
        axes[1, 0].hist(self.energy_history[len(self.energy_history)//2:], 
                       bins=50, density=True, alpha=0.7)
        axes[1, 0].set_title('エネルギーの分布（後半のみ）')
        axes[1, 0].set_xlabel('エネルギー')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 磁化の分布
        axes[1, 1].hist(self.magnetization_history[len(self.magnetization_history)//2:], 
                       bins=50, density=True, alpha=0.7)
        axes[1, 1].set_title('磁化の分布（後半のみ）')
        axes[1, 1].set_xlabel('磁化')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 受容率の時系列
        axes[1, 2].plot(steps, self.acceptance_history)
        axes[1, 2].set_title('受容率の時間発展')
        axes[1, 2].set_xlabel('MCステップ')
        axes[1, 2].set_ylabel('受容率')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 温度依存性の調査
def temperature_dependence_study():
    """温度依存性の調査"""
    
    size = 20
    temperatures = np.linspace(1.5, 3.5, 20)
    n_steps = 5000
    
    energies = []
    magnetizations = []
    specific_heats = []
    susceptibilities = []
    
    print("=== イジングモデルの温度依存性調査 ===")
    
    for i, T in enumerate(temperatures):
        print(f"温度 {T:.3f} ({i+1}/{len(temperatures)})")
        
        model = IsingModel2D(size=size, temperature=T)
        model.simulate(n_steps, record_interval=10)
        
        # 平衡状態のデータのみ使用（後半）
        equilibrium_start = len(model.energy_history) // 2
        
        E = np.array(model.energy_history[equilibrium_start:])
        M = np.array(model.magnetization_history[equilibrium_start:])
        
        # 統計量の計算
        mean_E = np.mean(E)
        mean_M = np.mean(np.abs(M))  # 磁化の絶対値
        
        # 比熱 C = (⟨E²⟩ - ⟨E⟩²) / T²
        C = (np.mean(E**2) - mean_E**2) / (T**2)
        
        # 磁化率 χ = (⟨M²⟩ - ⟨M⟩²) / T
        chi = (np.mean(M**2) - np.mean(M)**2) / T
        
        energies.append(mean_E / (size * size))  # 1スピンあたりのエネルギー
        magnetizations.append(mean_M / (size * size))  # 1スピンあたりの磁化
        specific_heats.append(C / (size * size))
        susceptibilities.append(chi / (size * size))
    
    # 結果の可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # エネルギー vs 温度
    axes[0, 0].plot(temperatures, energies, 'o-')
    axes[0, 0].set_xlabel('温度')
    axes[0, 0].set_ylabel('1スピンあたりのエネルギー')
    axes[0, 0].set_title('エネルギー vs 温度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 磁化 vs 温度
    axes[0, 1].plot(temperatures, magnetizations, 'o-', color='red')
    axes[0, 1].axvline(2.269, color='k', linestyle='--', alpha=0.7, label='理論臨界温度')
    axes[0, 1].set_xlabel('温度')
    axes[0, 1].set_ylabel('1スピンあたりの磁化')
    axes[0, 1].set_title('磁化 vs 温度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 比熱 vs 温度
    axes[1, 0].plot(temperatures, specific_heats, 'o-', color='green')
    axes[1, 0].axvline(2.269, color='k', linestyle='--', alpha=0.7, label='理論臨界温度')
    axes[1, 0].set_xlabel('温度')
    axes[1, 0].set_ylabel('比熱')
    axes[1, 0].set_title('比熱 vs 温度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 磁化率 vs 温度
    axes[1, 1].plot(temperatures, susceptibilities, 'o-', color='purple')
    axes[1, 1].axvline(2.269, color='k', linestyle='--', alpha=0.7, label='理論臨界温度')
    axes[1, 1].set_xlabel('温度')
    axes[1, 1].set_ylabel('磁化率')
    axes[1, 1].set_title('磁化率 vs 温度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return temperatures, energies, magnetizations, specific_heats, susceptibilities

# 個別シミュレーションの実行例
def ising_demo():
    """イジングモデルのデモンストレーション"""
    
    # 異なる温度での比較
    temperatures = [1.5, 2.269, 3.0]  # 低温、臨界温度、高温
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, T in enumerate(temperatures):
        print(f"\n温度 T = {T}")
        
        model = IsingModel2D(size=30, temperature=T)
        model.simulate(n_steps=3000, record_interval=10)
        
        # スピン配置の表示
        im = axes[i].imshow(model.spins, cmap='RdBu', vmin=-1, vmax=1)
        axes[i].set_title(f'T = {T:.3f}')
        axes[i].set_xlabel('x')
        if i == 0:
            axes[i].set_ylabel('y')
        
        # 統計量の表示
        final_magnetization = model.magnetization() / (model.size * model.size)
        final_energy = model.total_energy() / (model.size * model.size)
        
        print(f"  最終磁化（1スピンあたり）: {final_magnetization:.3f}")
        print(f"  最終エネルギー（1スピンあたり）: {final_energy:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    # 詳細な解析（臨界温度付近）
    print(f"\n詳細解析：臨界温度付近 (T = 2.269)")
    critical_model = IsingModel2D(size=25, temperature=2.269)
    critical_model.simulate(n_steps=5000, record_interval=5)
    critical_model.plot_results()
    
    return critical_model

# デモ実行
print("イジングモデルのシミュレーション...")
ising_model = ising_demo()

print("\n温度依存性の調査...")
temp_data = temperature_dependence_study()
```

この章では、MCMCを実際の問題に適用する実践的なワークフローを詳しく解説しました。次章では、これらの手法で得られた結果の信頼性を評価する収束診断と性能評価の手法を学びます。

---

## 第5章: 収束診断と性能評価

### 5.1 収束診断の重要性

MCMCは理論的には無限回のイテレーションで目標分布に収束することが保証されていますが、実際の計算では有限回で打ち切る必要があります。そのため、**「得られたサンプルが本当に目標分布から来ているか」**を慎重に検証することが不可欠です。

#### 5.1.1 収束診断が必要な理由

1. **初期値の影響**: マルコフ連鎖の初期値が目標分布から大きく外れている場合、定常分布に到達するまで時間がかかる
2. **局所最適解への囚われ**: 多峰性のある分布では、チェーンが一つのモードに留まり、他の領域を探索できない可能性
3. **収束の遅さ**: パラメータ間の強い相関や提案分布の不適切な設定により、収束が著しく遅くなる場合
4. **実装上のバグ**: アルゴリズムの実装ミスや数値計算エラー

#### 5.1.2 診断手法の分類

収束診断は大きく以下に分類されます：

- **視覚的診断**: トレースプロット、自己相関関数、ランニング平均など
- **定量的診断**: Gelman-Rubin統計量、実効サンプルサイズ、Geweke診断など
- **複数チェーン診断**: 異なる初期値から開始した複数のチェーンの比較

### 5.2 視覚的診断手法

#### 5.2.1 包括的診断プロット関数

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from statsmodels.tsa.stattools import acf

class MCMCDiagnostics:
    def __init__(self, samples, parameter_names=None, true_values=None):
        """
        MCMC収束診断クラス
        
        Parameters:
        -----------
        samples : array-like, shape (n_samples, n_params) or (n_samples,)
            MCMCサンプル
        parameter_names : list of str, optional
            パラメータ名のリスト
        true_values : array-like, optional
            パラメータの真値（分かっている場合）
        """
        self.samples = np.atleast_2d(samples)
        if self.samples.shape[0] == 1:
            self.samples = self.samples.T
        
        self.n_samples, self.n_params = self.samples.shape
        
        if parameter_names is None:
            self.parameter_names = [f'θ_{i+1}' for i in range(self.n_params)]
        else:
            self.parameter_names = parameter_names
        
        self.true_values = true_values
    
    def trace_plots(self, figsize=None, max_plots_per_row=3):
        """トレースプロットの作成"""
        if figsize is None:
            figsize = (15, 4 * ((self.n_params - 1) // max_plots_per_row + 1))
        
        n_rows = (self.n_params - 1) // max_plots_per_row + 1
        fig, axes = plt.subplots(n_rows, max_plots_per_row, figsize=figsize)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.n_params):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            
            axes[row, col].plot(self.samples[:, i], alpha=0.8)
            
            if self.true_values is not None:
                axes[row, col].axhline(self.true_values[i], color='red', 
                                      linestyle='--', linewidth=2, label='真値')
                axes[row, col].legend()
            
            axes[row, col].set_title(f'{self.parameter_names[i]} のトレース')
            axes[row, col].set_xlabel('イテレーション')
            axes[row, col].set_ylabel(self.parameter_names[i])
            axes[row, col].grid(True, alpha=0.3)
        
        # 空のプロットを非表示
        for i in range(self.n_params, n_rows * max_plots_per_row):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def running_mean_plots(self, figsize=None, max_plots_per_row=3):
        """累積平均の収束プロット"""
        if figsize is None:
            figsize = (15, 4 * ((self.n_params - 1) // max_plots_per_row + 1))
        
        n_rows = (self.n_params - 1) // max_plots_per_row + 1
        fig, axes = plt.subplots(n_rows, max_plots_per_row, figsize=figsize)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.n_params):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            
            # 累積平均の計算
            cumulative_mean = np.cumsum(self.samples[:, i]) / np.arange(1, self.n_samples + 1)
            
            axes[row, col].plot(cumulative_mean)
            
            if self.true_values is not None:
                axes[row, col].axhline(self.true_values[i], color='red', 
                                      linestyle='--', linewidth=2, label='真値')
                axes[row, col].legend()
            
            # 最終的な平均値
            final_mean = cumulative_mean[-1]
            axes[row, col].axhline(final_mean, color='green', 
                                  linestyle=':', alpha=0.7, 
                                  label=f'最終平均: {final_mean:.4f}')
            axes[row, col].legend()
            
            axes[row, col].set_title(f'{self.parameter_names[i]} の累積平均')
            axes[row, col].set_xlabel('イテレーション')
            axes[row, col].set_ylabel('累積平均')
            axes[row, col].grid(True, alpha=0.3)
        
        # 空のプロットを非表示
        for i in range(self.n_params, n_rows * max_plots_per_row):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def autocorr_plots(self, max_lags=50, figsize=None, max_plots_per_row=3):
        """自己相関プロット"""
        if figsize is None:
            figsize = (15, 4 * ((self.n_params - 1) // max_plots_per_row + 1))
        
        n_rows = (self.n_params - 1) // max_plots_per_row + 1
        fig, axes = plt.subplots(n_rows, max_plots_per_row, figsize=figsize)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.n_params):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            
            # 自己相関の計算
            try:
                autocorr = acf(self.samples[:, i], nlags=max_lags, fft=True)
                lags = np.arange(len(autocorr))
                
                axes[row, col].plot(lags, autocorr, 'b-', alpha=0.8)
                axes[row, col].axhline(0, color='k', linestyle='--', alpha=0.5)
                
                # 5%有意水準のライン
                n = len(self.samples[:, i])
                conf_int = 1.96 / np.sqrt(n)
                axes[row, col].axhline(conf_int, color='r', linestyle=':', alpha=0.7)
                axes[row, col].axhline(-conf_int, color='r', linestyle=':', alpha=0.7)
                
            except Exception as e:
                print(f"自己相関の計算でエラー (パラメータ {i}): {e}")
                axes[row, col].text(0.5, 0.5, 'エラー', transform=axes[row, col].transAxes,
                                   ha='center', va='center')
            
            axes[row, col].set_title(f'{self.parameter_names[i]} の自己相関')
            axes[row, col].set_xlabel('ラグ')
            axes[row, col].set_ylabel('自己相関')
            axes[row, col].grid(True, alpha=0.3)
        
        # 空のプロットを非表示
        for i in range(self.n_params, n_rows * max_plots_per_row):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def posterior_plots(self, bins=50, figsize=None, max_plots_per_row=3):
        """事後分布のヒストグラム"""
        if figsize is None:
            figsize = (15, 4 * ((self.n_params - 1) // max_plots_per_row + 1))
        
        n_rows = (self.n_params - 1) // max_plots_per_row + 1
        fig, axes = plt.subplots(n_rows, max_plots_per_row, figsize=figsize)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.n_params):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            
            axes[row, col].hist(self.samples[:, i], bins=bins, density=True, 
                               alpha=0.7, edgecolor='black')
            
            if self.true_values is not None:
                axes[row, col].axvline(self.true_values[i], color='red', 
                                      linestyle='--', linewidth=2, label='真値')
                axes[row, col].legend()
            
            # 統計量の表示
            mean_val = np.mean(self.samples[:, i])
            std_val = np.std(self.samples[:, i])
            axes[row, col].axvline(mean_val, color='blue', 
                                  linestyle=':', alpha=0.7, 
                                  label=f'平均: {mean_val:.4f}')
            
            axes[row, col].set_title(f'{self.parameter_names[i]} の事後分布\n'
                                   f'平均={mean_val:.4f}, 標準偏差={std_val:.4f}')
            axes[row, col].set_xlabel(self.parameter_names[i])
            axes[row, col].set_ylabel('密度')
            axes[row, col].grid(True, alpha=0.3)
        
        # 空のプロットを非表示
        for i in range(self.n_params, n_rows * max_plots_per_row):
            row = i // max_plots_per_row
            col = i % max_plots_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_diagnostics(self):
        """包括的な診断プロットの作成"""
        print("=== MCMC収束診断 ===")
        print(f"サンプル数: {self.n_samples}")
        print(f"パラメータ数: {self.n_params}")
        
        # 基本統計量
        print("\n基本統計量:")
        for i, name in enumerate(self.parameter_names):
            mean_val = np.mean(self.samples[:, i])
            std_val = np.std(self.samples[:, i])
            ci_lower = np.percentile(self.samples[:, i], 2.5)
            ci_upper = np.percentile(self.samples[:, i], 97.5)
            
            print(f"  {name}: 平均={mean_val:8.4f}, 標準偏差={std_val:8.4f}, "
                  f"95%CI=[{ci_lower:8.4f}, {ci_upper:8.4f}]")
            
            if self.true_values is not None:
                bias = mean_val - self.true_values[i]
                print(f"         真値={self.true_values[i]:8.4f}, バイアス={bias:8.4f}")
        
        # プロットの作成
        print("\nトレースプロット:")
        self.trace_plots()
        
        print("累積平均プロット:")
        self.running_mean_plots()
        
        print("自己相関プロット:")
        self.autocorr_plots()
        
        print("事後分布プロット:")
        self.posterior_plots()

# 使用例
def diagnostics_demo():
    """診断手法のデモンストレーション"""
    
    # 合成データでテスト（第4章のベイズ線形回帰の結果を使用）
    np.random.seed(42)
    
    # 真のパラメータ
    true_beta = [2.0, -1.5, 0.8]
    true_sigma2 = 0.25
    
    # 模擬的なMCMCサンプル（実際には第4章のBayesianLinearRegressionから取得）
    n_samples = 8000
    
    # 真の値周辺の正規分布からサンプル生成（MCMCの結果を模擬）
    beta_samples = []
    for i, true_val in enumerate(true_beta):
        samples = np.random.normal(true_val, 0.1, n_samples)
        # 自己相関を追加（現実的なMCMCサンプルをシミュレート）
        for j in range(1, len(samples)):
            samples[j] = 0.3 * samples[j-1] + 0.7 * samples[j]
        beta_samples.append(samples)
    
    sigma2_samples = np.random.gamma(10, true_sigma2/10, n_samples)
    
    # 全パラメータをまとめる
    all_samples = np.column_stack(beta_samples + [sigma2_samples])
    parameter_names = ['β₀', 'β₁', 'β₂', 'σ²']
    true_values = true_beta + [true_sigma2]
    
    # 診断の実行
    diagnostics = MCMCDiagnostics(all_samples, parameter_names, true_values)
    diagnostics.comprehensive_diagnostics()
    
    return diagnostics

# デモ実行
diagnostics_result = diagnostics_demo()
```

### 5.3 定量的診断指標

#### 5.3.1 Gelman-Rubin診断（R̂統計量）

Gelman-Rubin診断は、複数の独立したチェーンを比較することで収束を評価する最も重要な手法の一つです。

```python
class GelmanRubinDiagnostic:
    def __init__(self, chains_list, parameter_names=None):
        """
        Gelman-Rubin診断
        
        Parameters:
        -----------
        chains_list : list of arrays
            各チェーンのサンプル [chain1, chain2, ...] 
            各chainはshape (n_samples, n_params)
        parameter_names : list of str, optional
            パラメータ名
        """
        self.chains_list = [np.atleast_2d(chain) for chain in chains_list]
        
        # すべてのチェーンが同じ形状であることを確認
        shapes = [chain.shape for chain in self.chains_list]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("すべてのチェーンは同じ形状である必要があります")
        
        self.n_chains = len(chains_list)
        self.n_samples, self.n_params = self.chains_list[0].shape
        
        if parameter_names is None:
            self.parameter_names = [f'θ_{i+1}' for i in range(self.n_params)]
        else:
            self.parameter_names = parameter_names
    
    def compute_rhat(self):
        """R̂統計量の計算"""
        rhat_values = []
        
        for param_idx in range(self.n_params):
            # 各チェーンからパラメータを抽出
            chains_param = [chain[:, param_idx] for chain in self.chains_list]
            
            # チェーン内分散の平均（W）
            chain_vars = [np.var(chain, ddof=1) for chain in chains_param]
            W = np.mean(chain_vars)
            
            # チェーン間分散（B）
            chain_means = [np.mean(chain) for chain in chains_param]
            overall_mean = np.mean(chain_means)
            B = self.n_samples * np.var(chain_means, ddof=1)
            
            # マージナル事後分散の推定値
            var_plus = ((self.n_samples - 1) * W + B) / self.n_samples
            
            # R̂統計量
            if W > 0:
                rhat = np.sqrt(var_plus / W)
            else:
                rhat = np.inf
            
            rhat_values.append(rhat)
        
        return np.array(rhat_values)
    
    def compute_neff(self):
        """実効サンプルサイズの計算"""
        neff_values = []
        
        for param_idx in range(self.n_params):
            chains_param = [chain[:, param_idx] for chain in self.chains_list]
            
            # 各チェーンの自己相関を計算
            autocorr_sums = []
            for chain in chains_param:
                try:
                    # statsmodelsのacfを使用
                    autocorr = acf(chain, nlags=min(200, len(chain)//4), fft=True)
                    # 最初に負になる点で打ち切り
                    cutoff = len(autocorr)
                    for i in range(1, len(autocorr)):
                        if autocorr[i] <= 0.05:  # 5%以下で打ち切り
                            cutoff = i
                            break
                    
                    # 積分自己相関時間
                    tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
                    autocorr_sums.append(tau_int)
                    
                except Exception:
                    # エラーの場合は保守的な値を使用
                    autocorr_sums.append(len(chain) / 10)
            
            # 平均自己相関時間
            avg_tau_int = np.mean(autocorr_sums)
            
            # 実効サンプルサイズ
            total_samples = self.n_chains * self.n_samples
            neff = total_samples / avg_tau_int if avg_tau_int > 0 else total_samples
            
            neff_values.append(neff)
        
        return np.array(neff_values)
    
    def summary(self):
        """診断結果の要約"""
        rhat_values = self.compute_rhat()
        neff_values = self.compute_neff()
        
        print("=== Gelman-Rubin診断結果 ===")
        print(f"チェーン数: {self.n_chains}")
        print(f"各チェーンのサンプル数: {self.n_samples}")
        print(f"総サンプル数: {self.n_chains * self.n_samples}")
        
        print(f"\n{'パラメータ':>10} {'R̂':>8} {'Neff':>8} {'判定':>8}")
        print("-" * 40)
        
        all_converged = True
        for i, name in enumerate(self.parameter_names):
            rhat = rhat_values[i]
            neff = neff_values[i]
            
            # 収束判定
            if rhat < 1.1 and neff > 400:
                status = "✓"
            elif rhat < 1.1 and neff > 100:
                status = "△"
                all_converged = False
            else:
                status = "✗"
                all_converged = False
            
            print(f"{name:>10} {rhat:8.4f} {neff:8.0f} {status:>8}")
        
        print("-" * 40)
        if all_converged:
            print("判定: すべてのパラメータが収束している")
        else:
            print("判定: 一部のパラメータで収束が不十分")
            print("  ✓: 良好 (R̂<1.1, Neff>400)")
            print("  △: 注意 (R̂<1.1, Neff>100)")
            print("  ✗: 不良 (R̂≥1.1 or Neff≤100)")
        
        return rhat_values, neff_values
    
    def plot_chains(self, figsize=None):
        """複数チェーンのトレースプロット"""
        if figsize is None:
            figsize = (15, 4 * self.n_params)
        
        fig, axes = plt.subplots(self.n_params, 1, figsize=figsize)
        if self.n_params == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for param_idx in range(self.n_params):
            for chain_idx, chain in enumerate(self.chains_list):
                color = colors[chain_idx % len(colors)]
                axes[param_idx].plot(chain[:, param_idx], 
                                   color=color, alpha=0.7, 
                                   label=f'Chain {chain_idx + 1}')
            
            axes[param_idx].set_title(f'{self.parameter_names[param_idx]} - 複数チェーン比較')
            axes[param_idx].set_xlabel('イテレーション')
            axes[param_idx].set_ylabel(self.parameter_names[param_idx])
            axes[param_idx].legend()
            axes[param_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 複数チェーンの生成とテスト
def multiple_chains_demo():
    """複数チェーンを使った診断のデモ"""
    
    np.random.seed(42)
    
    # 真のパラメータ
    true_params = [2.0, -1.5]
    
    # 異なる初期値から開始した3つのチェーンを生成
    n_samples = 5000
    n_chains = 4
    
    chains = []
    initial_values = [[-2, 3], [0, 0], [4, -3], [1, -2]]  # 異なる初期値
    
    for i in range(n_chains):
        print(f"チェーン {i+1} 生成中... (初期値: {initial_values[i]})")
        
        # 各チェーンを異なる自己相関で生成
        chain_samples = []
        for j, true_val in enumerate(true_params):
            # 初期値から真値に向かって収束するサンプル
            samples = np.zeros(n_samples)
            samples[0] = initial_values[i][j]
            
            for t in range(1, n_samples):
                # AR(1)プロセス + ドリフト
                drift = 0.01 * (true_val - samples[t-1])
                samples[t] = samples[t-1] + drift + np.random.normal(0, 0.1)
                
                # 収束後は真値周辺でランダムウォーク
                if t > 1000:
                    samples[t] = true_val + np.random.normal(0, 0.15)
                    if t > 1001:
                        samples[t] = 0.3 * samples[t-1] + 0.7 * samples[t]
            
            chain_samples.append(samples)
        
        chains.append(np.column_stack(chain_samples))
    
    # Gelman-Rubin診断の実行
    parameter_names = ['θ₁', 'θ₂']
    gr_diagnostic = GelmanRubinDiagnostic(chains, parameter_names)
    
    # チェーンの可視化
    gr_diagnostic.plot_chains()
    
    # 診断結果
    rhat_values, neff_values = gr_diagnostic.summary()
    
    return gr_diagnostic, chains

# デモ実行
gr_result, chains_data = multiple_chains_demo()
```

#### 5.3.2 実効サンプルサイズ（ESS）の詳細計算

```python
def compute_effective_sample_size(samples, method='autocorr'):
    """
    実効サンプルサイズの計算
    
    Parameters:
    -----------
    samples : array-like
        MCMCサンプル
    method : str
        計算方法 ('autocorr', 'variogram', 'batch')
    
    Returns:
    --------
    ess : float
        実効サンプルサイズ
    """
    samples = np.asarray(samples)
    n = len(samples)
    
    if method == 'autocorr':
        # 自己相関関数による方法
        try:
            # 自己相関の計算
            autocorr = acf(samples, nlags=min(n//4, 200), fft=True)
            
            # 最初に負になる点または十分小さくなる点で打ち切り
            cutoff = len(autocorr)
            for i in range(1, len(autocorr)):
                if autocorr[i] <= 0.05:
                    cutoff = i
                    break
            
            # 積分自己相関時間
            tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
            
            # 実効サンプルサイズ
            ess = n / tau_int
            
        except Exception:
            ess = n / 10  # 保守的な値
    
    elif method == 'variogram':
        # バリオグラム法
        def variogram(lag):
            if lag >= n:
                return 0
            diff = samples[lag:] - samples[:-lag]
            return 0.5 * np.mean(diff**2)
        
        # ラグ0での分散
        var_0 = variogram(0)
        if var_0 == 0:
            return n
        
        # 様々なラグでのバリオグラム
        lags = range(1, min(n//4, 100))
        variograms = [variogram(lag) for lag in lags]
        
        # プラトーに達するラグを探す
        plateau_value = 2 * np.var(samples)
        tau_int = 1
        for i, vg in enumerate(variograms):
            if vg >= 0.95 * plateau_value:
                tau_int = lags[i]
                break
        
        ess = n / tau_int
    
    elif method == 'batch':
        # バッチ平均法
        batch_sizes = [10, 20, 50, 100]
        ess_estimates = []
        
        for batch_size in batch_sizes:
            if n < batch_size * 10:
                continue
            
            n_batches = n // batch_size
            batches = samples[:n_batches * batch_size].reshape(n_batches, batch_size)
            batch_means = np.mean(batches, axis=1)
            
            # バッチ平均の分散
            batch_var = np.var(batch_means, ddof=1)
            sample_var = np.var(samples, ddof=1)
            
            if sample_var > 0:
                ess_est = n * batch_var / sample_var
                ess_estimates.append(ess_est)
        
        ess = np.median(ess_estimates) if ess_estimates else n / 10
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max(1, min(ess, n))  # 1以上n以下にクリップ

# ESS計算のデモ
def ess_comparison_demo():
    """異なる手法でのESS計算比較"""
    
    np.random.seed(42)
    n_samples = 10000
    
    # 異なる自己相関を持つサンプルを生成
    correlations = [0.0, 0.3, 0.7, 0.9]
    
    results = []
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, rho in enumerate(correlations):
        # AR(1)プロセスでサンプル生成
        samples = np.zeros(n_samples)
        samples[0] = np.random.normal()
        
        for t in range(1, n_samples):
            samples[t] = rho * samples[t-1] + np.sqrt(1 - rho**2) * np.random.normal()
        
        # 各手法でESSを計算
        ess_autocorr = compute_effective_sample_size(samples, 'autocorr')
        ess_variogram = compute_effective_sample_size(samples, 'variogram')
        ess_batch = compute_effective_sample_size(samples, 'batch')
        
        # 理論値
        theoretical_ess = n_samples * (1 - rho) / (1 + rho) if rho < 1 else 1
        
        results.append({
            'correlation': rho,
            'autocorr': ess_autocorr,
            'variogram': ess_variogram,
            'batch': ess_batch,
            'theoretical': theoretical_ess
        })
        
        # トレースプロット
        axes[i].plot(samples[:1000])
        axes[i].set_title(f'ρ = {rho:.1f}\n'
                         f'ESS: autocorr={ess_autocorr:.0f}, '
                         f'理論値={theoretical_ess:.0f}')
        axes[i].set_xlabel('イテレーション')
        axes[i].set_ylabel('値')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print("=== ESS計算手法の比較 ===")
    print(f"{'相関':>6} {'自己相関法':>10} {'バリオグラム法':>12} {'バッチ法':>10} {'理論値':>8} {'効率':>6}")
    print("-" * 60)
    
    for result in results:
        rho = result['correlation']
        ess_auto = result['autocorr']
        ess_vario = result['variogram']
        ess_batch = result['batch']
        theoretical = result['theoretical']
        efficiency = ess_auto / n_samples * 100
        
        print(f"{rho:6.1f} {ess_auto:10.0f} {ess_vario:12.0f} {ess_batch:10.0f} "
              f"{theoretical:8.0f} {efficiency:5.1f}%")
    
    return results

# デモ実行
ess_results = ess_comparison_demo()
```

### 5.4 収束の加速化技法

#### 5.4.1 適応的MCMC

```python
class AdaptiveMetropolisHastings:
    def __init__(self, target_logpdf, initial_value, target_acceptance_rate=0.44):
        """
        適応的メトロポリス・ヘイスティングス法
        
        Parameters:
        -----------
        target_logpdf : callable
            目標分布の対数確率密度関数
        initial_value : array-like
            初期値
        target_acceptance_rate : float
            目標受容率
        """
        self.target_logpdf = target_logpdf
        self.current_value = np.array(initial_value)
        self.dimension = len(self.current_value)
        self.target_acceptance_rate = target_acceptance_rate
        
        # 適応パラメータ
        self.covariance = np.eye(self.dimension)
        self.mean = self.current_value.copy()
        self.n_iterations = 0
        self.n_accepted = 0
        
        # 履歴
        self.samples = []
        self.acceptance_history = []
        self.covariance_history = []
        
        # 適応化パラメータ
        self.adaptation_interval = 100
        self.epsilon = 1e-8  # 数値安定性のため
        
    def update_covariance(self, new_sample):
        """共分散行列の適応的更新"""
        self.n_iterations += 1
        
        # オンライン平均・共分散更新（Welford's algorithm）
        delta = new_sample - self.mean
        self.mean += delta / self.n_iterations
        
        if self.n_iterations > 1:
            delta2 = new_sample - self.mean
            # 外積を加算
            outer_product = np.outer(delta, delta2)
            self.covariance = ((self.n_iterations - 2) * self.covariance + 
                              outer_product) / (self.n_iterations - 1)
        
        # 数値安定性のために最小固有値を保証
        eigenvals = np.linalg.eigvals(self.covariance)
        if np.min(eigenvals) < self.epsilon:
            self.covariance += self.epsilon * np.eye(self.dimension)
    
    def adapt_proposal_scale(self):
        """提案分布のスケール適応"""
        if self.n_iterations % self.adaptation_interval == 0 and self.n_iterations > 0:
            current_acceptance_rate = self.n_accepted / self.adaptation_interval
            
            # スケール調整（Robbins-Monro型）
            adaptation_rate = min(0.01, 1.0 / np.sqrt(self.n_iterations))
            
            if current_acceptance_rate > self.target_acceptance_rate:
                # 受容率が高すぎる場合、提案分布を拡大
                scale_factor = np.exp(adaptation_rate)
            else:
                # 受容率が低すぎる場合、提案分布を縮小
                scale_factor = np.exp(-adaptation_rate)
            
            self.covariance *= scale_factor
            
            # 受容率をリセット
            self.acceptance_history.append(current_acceptance_rate)
            self.n_accepted = 0
    
    def sample(self, n_samples, burn_in=1000):
        """適応的サンプリング"""
        print(f"適応的MCMC開始... (burn_in={burn_in}, samples={n_samples})")
        
        current_logp = self.target_logpdf(self.current_value)
        total_iterations = burn_in + n_samples
        
        for iteration in range(total_iterations):
            # 提案
            try:
                proposal = np.random.multivariate_normal(
                    self.current_value, 
                    2.38**2 / self.dimension * self.covariance  # 最適スケーリング
                )
            except np.linalg.LinAlgError:
                # 共分散行列が特異な場合
                proposal = self.current_value + np.random.normal(0, 0.1, self.dimension)
            
            proposal_logp = self.target_logpdf(proposal)
            
            # 受容確率
            log_alpha = min(0, proposal_logp - current_logp)
            
            # 受容判定
            if np.log(np.random.random()) < log_alpha:
                self.current_value = proposal
                current_logp = proposal_logp
                self.n_accepted += 1
            
            # 共分散行列の更新
            self.update_covariance(self.current_value)
            
            # スケール適応
            self.adapt_proposal_scale()
            
            # サンプルの保存（バーンイン後）
            if iteration >= burn_in:
                self.samples.append(self.current_value.copy())
            
            # 共分散履歴の保存（診断用）
            if iteration % 100 == 0:
                self.covariance_history.append(self.covariance.copy())
            
            # 進捗表示
            if (iteration + 1) % (total_iterations // 10) == 0:
                acceptance_rate = self.n_accepted / min(iteration + 1, self.adaptation_interval)
                print(f"  進捗: {(iteration + 1) / total_iterations * 100:.1f}%, "
                      f"受容率: {acceptance_rate:.3f}")
        
        self.samples = np.array(self.samples)
        print("適応的MCMC完了!")
        return self.samples
    
    def plot_adaptation_diagnostics(self):
        """適応化の診断プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 受容率の推移
        if self.acceptance_history:
            axes[0, 0].plot(self.acceptance_history)
            axes[0, 0].axhline(self.target_acceptance_rate, color='r', 
                              linestyle='--', label=f'目標受容率: {self.target_acceptance_rate}')
            axes[0, 0].set_title('受容率の推移')
            axes[0, 0].set_xlabel('適応インターバル')
            axes[0, 0].set_ylabel('受容率')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 共分散行列の対角成分の推移
        if self.covariance_history:
            cov_diagonals = [np.diag(cov) for cov in self.covariance_history]
            cov_diagonals = np.array(cov_diagonals)
            
            for i in range(self.dimension):
                axes[0, 1].plot(cov_diagonals[:, i], label=f'次元 {i+1}')
            
            axes[0, 1].set_title('共分散行列対角成分の推移')
            axes[0, 1].set_xlabel('適応ステップ (×100)')
            axes[0, 1].set_ylabel('分散')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # サンプルのトレース
        if len(self.samples) > 0:
            for i in range(min(self.dimension, 2)):
                axes[1, i].plot(self.samples[:, i])
                axes[1, i].set_title(f'パラメータ {i+1} のトレース')
                axes[1, i].set_xlabel('イテレーション')
                axes[1, i].set_ylabel(f'θ_{i+1}')
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 適応的MCMCのデモ
def adaptive_mcmc_demo():
    """適応的MCMCのデモンストレーション"""
    
    # 目標分布：相関のある2次元正規分布
    mu = np.array([2.0, -1.0])
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.5]])
    
    inv_Sigma = np.linalg.inv(Sigma)
    
    def target_logpdf(x):
        """目標分布の対数確率密度"""
        diff = x - mu
        return -0.5 * diff @ inv_Sigma @ diff
    
    # 適応的MCMC
    initial_value = [0.0, 0.0]  # 真値から離れた初期値
    adaptive_sampler = AdaptiveMetropolisHastings(target_logpdf, initial_value)
    adaptive_samples = adaptive_sampler.sample(n_samples=8000, burn_in=2000)
    
    # 通常のMH法との比較
    from scipy.stats import multivariate_normal
    
    # 通常のMH法（固定提案分布）
    def fixed_mh_sampling(n_samples, burn_in=2000):
        samples = []
        current = np.array([0.0, 0.0])
        current_logp = target_logpdf(current)
        n_accepted = 0
        
        # 固定の提案共分散（不適切に設定）
        proposal_cov = np.array([[0.1, 0.0], [0.0, 0.1]])
        
        for i in range(burn_in + n_samples):
            proposal = np.random.multivariate_normal(current, proposal_cov)
            proposal_logp = target_logpdf(proposal)
            
            if np.log(np.random.random()) < min(0, proposal_logp - current_logp):
                current = proposal
                current_logp = proposal_logp
                n_accepted += 1
            
            if i >= burn_in:
                samples.append(current.copy())
        
        print(f"固定MH法の受容率: {n_accepted / (burn_in + n_samples):.3f}")
        return np.array(samples)
    
    fixed_samples = fixed_mh_sampling(8000)
    
    # 真の分布からのサンプル
    true_samples = np.random.multivariate_normal(mu, Sigma, 8000)
    
    # 結果の比較
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 散布図
    methods = ['適応的MCMC', '固定MH法', '真の分布']
    sample_sets = [adaptive_samples, fixed_samples, true_samples]
    colors = ['blue', 'red', 'green']
    
    for i, (method, samples, color) in enumerate(zip(methods, sample_sets, colors)):
        axes[0, i].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, color=color)
        axes[0, i].set_title(f'{method}\n平均: [{np.mean(samples[:, 0]):.3f}, {np.mean(samples[:, 1]):.3f}]')
        axes[0, i].set_xlabel('θ₁')
        axes[0, i].set_ylabel('θ₂')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')
    
    # トレースプロット比較
    axes[1, 0].plot(adaptive_samples[:2000, 0], color='blue', alpha=0.7, label='適応的')
    axes[1, 0].plot(fixed_samples[:2000, 0], color='red', alpha=0.7, label='固定')
    axes[1, 0].axhline(mu[0], color='black', linestyle='--', label='真値')
    axes[1, 0].set_title('θ₁のトレース比較')
    axes[1, 0].set_xlabel('イテレーション')
    axes[1, 0].set_ylabel('θ₁')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(adaptive_samples[:2000, 1], color='blue', alpha=0.7, label='適応的')
    axes[1, 1].plot(fixed_samples[:2000, 1], color='red', alpha=0.7, label='固定')
    axes[1, 1].axhline(mu[1], color='black', linestyle='--', label='真値')
    axes[1, 1].set_title('θ₂のトレース比較')
    axes[1, 1].set_xlabel('イテレーション')
    axes[1, 1].set_ylabel('θ₂')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # ESS比較
    ess_adaptive = [compute_effective_sample_size(adaptive_samples[:, i]) for i in range(2)]
    ess_fixed = [compute_effective_sample_size(fixed_samples[:, i]) for i in range(2)]
    
    x_pos = np.arange(2)
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, ess_adaptive, width, label='適応的MCMC', color='blue', alpha=0.7)
    axes[1, 2].bar(x_pos + width/2, ess_fixed, width, label='固定MH法', color='red', alpha=0.7)
    axes[1, 2].set_xlabel('パラメータ')
    axes[1, 2].set_ylabel('実効サンプルサイズ')
    axes[1, 2].set_title('サンプリング効率の比較')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(['θ₁', 'θ₂'])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 適応化診断
    adaptive_sampler.plot_adaptation_diagnostics()
    
    # 数値比較
    print("=== サンプリング効率比較 ===")
    print(f"適応的MCMC - ESS: θ₁={ess_adaptive[0]:.0f}, θ₂={ess_adaptive[1]:.0f}")
    print(f"固定MH法    - ESS: θ₁={ess_fixed[0]:.0f}, θ₂={ess_fixed[1]:.0f}")
    print(f"効率比: θ₁={ess_adaptive[0]/ess_fixed[0]:.2f}x, θ₂={ess_adaptive[1]/ess_fixed[1]:.2f}x")
    
    return adaptive_sampler, adaptive_samples, fixed_samples

# デモ実行
adaptive_result = adaptive_mcmc_demo()
```

### 5.5 実践的な収束診断ワークフロー

#### 5.5.1 完全な診断パイプライン

```python
class ComprehensiveMCMCDiagnostics:
    def __init__(self, chains_list, parameter_names=None, true_values=None):
        """
        包括的MCMC診断クラス
        
        Parameters:
        -----------
        chains_list : list of arrays or single array
            MCMCチェーン（複数可）
        parameter_names : list of str, optional
            パラメータ名
        true_values : array-like, optional
            真値（既知の場合）
        """
        # 単一チェーンの場合はリストに変換
        if not isinstance(chains_list, list):
            chains_list = [chains_list]
        
        self.chains_list = [np.atleast_2d(chain) for chain in chains_list]
        self.n_chains = len(self.chains_list)
        self.n_samples, self.n_params = self.chains_list[0].shape
        
        if parameter_names is None:
            self.parameter_names = [f'θ_{i+1}' for i in range(self.n_params)]
        else:
            self.parameter_names = parameter_names
        
        self.true_values = true_values
    
    def run_all_diagnostics(self, output_file=None):
        """すべての診断を実行"""
        print("=" * 60)
        print("MCMC収束診断レポート")
        print("=" * 60)
        
        results = {}
        
        # 基本情報
        print(f"\n基本情報:")
        print(f"  チェーン数: {self.n_chains}")
        print(f"  各チェーンのサンプル数: {self.n_samples}")
        print(f"  パラメータ数: {self.n_params}")
        print(f"  総サンプル数: {self.n_chains * self.n_samples}")
        
        # 1. 基本統計量
        print(f"\n1. 基本統計量:")
        basic_stats = self._compute_basic_statistics()
        results['basic_stats'] = basic_stats
        
        # 2. Gelman-Rubin診断（複数チェーンの場合）
        if self.n_chains > 1:
            print(f"\n2. Gelman-Rubin診断:")
            gr_results = self._compute_gelman_rubin()
            results['gelman_rubin'] = gr_results
        else:
            print(f"\n2. Gelman-Rubin診断: スキップ（単一チェーン）")
        
        # 3. 実効サンプルサイズ
        print(f"\n3. 実効サンプルサイズ:")
        ess_results = self._compute_effective_sample_sizes()
        results['ess'] = ess_results
        
        # 4. 自己相関分析
        print(f"\n4. 自己相関分析:")
        autocorr_results = self._analyze_autocorrelation()
        results['autocorr'] = autocorr_results
        
        # 5. 収束判定
        print(f"\n5. 総合収束判定:")
        convergence_assessment = self._assess_convergence(results)
        results['convergence'] = convergence_assessment
        
        # 6. 推奨事項
        print(f"\n6. 推奨事項:")
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        # 結果の保存
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _compute_basic_statistics(self):
        """基本統計量の計算"""
        # 全チェーンを結合
        all_samples = np.vstack(self.chains_list)
        
        stats_results = {}
        
        print(f"{'パラメータ':>10} {'平均':>10} {'標準偏差':>10} {'95%CI下限':>10} {'95%CI上限':>10}")
        print("-" * 55)
        
        for i, name in enumerate(self.parameter_names):
            samples_i = all_samples[:, i]
            
            mean_val = np.mean(samples_i)
            std_val = np.std(samples_i, ddof=1)
            ci_lower = np.percentile(samples_i, 2.5)
            ci_upper = np.percentile(samples_i, 97.5)
            
            print(f"{name:>10} {mean_val:10.4f} {std_val:10.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            
            stats_results[name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
            if self.true_values is not None:
                bias = mean_val - self.true_values[i]
                coverage = ci_lower <= self.true_values[i] <= ci_upper
                stats_results[name].update({
                    'true_value': self.true_values[i],
                    'bias': bias,
                    'coverage': coverage
                })
        
        return stats_results
    
    def _compute_gelman_rubin(self):
        """Gelman-Rubin診断"""
        rhat_values = []
        
        for param_idx in range(self.n_params):
            chains_param = [chain[:, param_idx] for chain in self.chains_list]
            
            # チェーン内分散の平均
            chain_vars = [np.var(chain, ddof=1) for chain in chains_param]
            W = np.mean(chain_vars)
            
            # チェーン間分散
            chain_means = [np.mean(chain) for chain in chains_param]
            overall_mean = np.mean(chain_means)
            B = self.n_samples * np.var(chain_means, ddof=1)
            
            # マージナル事後分散の推定値
            var_plus = ((self.n_samples - 1) * W + B) / self.n_samples
            
            # R̂統計量
            rhat = np.sqrt(var_plus / W) if W > 0 else np.inf
            rhat_values.append(rhat)
        
        print(f"{'パラメータ':>10} {'R̂':>8} {'判定':>8}")
        print("-" * 30)
        
        gr_results = {}
        for i, (name, rhat) in enumerate(zip(self.parameter_names, rhat_values)):
            status = "✓" if rhat < 1.1 else "✗"
            print(f"{name:>10} {rhat:8.4f} {status:>8}")
            
            gr_results[name] = {
                'rhat': rhat,
                'converged': rhat < 1.1
            }
        
        return gr_results
    
    def _compute_effective_sample_sizes(self):
        """実効サンプルサイズの計算"""
        ess_results = {}
        
        print(f"{'パラメータ':>10} {'ESS':>8} {'効率':>8} {'判定':>8}")
        print("-" * 40)
        
        total_samples = self.n_chains * self.n_samples
        
        for i, name in enumerate(self.parameter_names):
            # 全チェーンを結合
            all_samples_i = np.hstack([chain[:, i] for chain in self.chains_list])
            
            ess = compute_effective_sample_size(all_samples_i)
            efficiency = ess / total_samples * 100
            
            # 判定基準
            if ess >= 400:
                status = "✓"
            elif ess >= 100:
                status = "△"
            else:
                status = "✗"
            
            print(f"{name:>10} {ess:8.0f} {efficiency:7.1f}% {status:>8}")
            
            ess_results[name] = {
                'ess': ess,
                'efficiency': efficiency,
                'adequate': ess >= 100
            }
        
        return ess_results
    
    def _analyze_autocorrelation(self):
        """自己相関分析"""
        autocorr_results = {}
        
        print(f"{'パラメータ':>10} {'積分時間':>10} {'判定':>8}")
        print("-" * 32)
        
        for i, name in enumerate(self.parameter_names):
            all_samples_i = np.hstack([chain[:, i] for chain in self.chains_list])
            
            try:
                autocorr = acf(all_samples_i, nlags=min(200, len(all_samples_i)//4), fft=True)
                
                # 積分自己相関時間
                cutoff = len(autocorr)
                for j in range(1, len(autocorr)):
                    if autocorr[j] <= 0.05:
                        cutoff = j
                        break
                
                tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
                
                # 判定
                status = "✓" if tau_int < 20 else "△" if tau_int < 50 else "✗"
                
                print(f"{name:>10} {tau_int:10.2f} {status:>8}")
                
                autocorr_results[name] = {
                    'tau_int': tau_int,
                    'low_autocorr': tau_int < 20
                }
                
            except Exception:
                print(f"{name:>10} {'エラー':>10} {'✗':>8}")
                autocorr_results[name] = {
                    'tau_int': float('inf'),
                    'low_autocorr': False
                }
        
        return autocorr_results
    
    def _assess_convergence(self, results):
        """総合的な収束判定"""
        convergence_issues = []
        
        # Gelman-Rubin判定
        if 'gelman_rubin' in results:
            for name, gr_result in results['gelman_rubin'].items():
                if not gr_result['converged']:
                    convergence_issues.append(f"{name}: R̂ = {gr_result['rhat']:.4f} > 1.1")
        
        # ESS判定
        for name, ess_result in results['ess'].items():
            if not ess_result['adequate']:
                convergence_issues.append(f"{name}: ESS = {ess_result['ess']:.0f} < 100")
        
        # 自己相関判定
        for name, autocorr_result in results['autocorr'].items():
            if not autocorr_result['low_autocorr']:
                tau_int = autocorr_result['tau_int']
                if not np.isinf(tau_int):
                    convergence_issues.append(f"{name}: 積分時間 = {tau_int:.2f} > 20")
        
        if not convergence_issues:
            print("すべてのパラメータが収束基準を満たしています ✓")
            overall_status = "converged"
        else:
            print("以下のパラメータで収束に問題があります:")
            for issue in convergence_issues:
                print(f"  - {issue}")
            overall_status = "not_converged"
        
        return {
            'overall_status': overall_status,
            'issues': convergence_issues
        }
    
    def _generate_recommendations(self, results):
        """推奨事項の生成"""
        recommendations = []
        
        convergence = results['convergence']
        
        if convergence['overall_status'] == 'converged':
            recommendations.append("✓ 現在のサンプリング設定は適切です")
            recommendations.append("✓ 分析を進めることができます")
        else:
            recommendations.append("改善が必要な項目:")
            
            # R̂に関する推奨
            if 'gelman_rubin' in results:
                high_rhat = [name for name, gr in results['gelman_rubin'].items() 
                           if not gr['converged']]
                if high_rhat:
                    recommendations.append(f"- より多くのイテレーションが必要: {', '.join(high_rhat)}")
                    recommendations.append("- 異なる初期値から複数チェーンを実行することを検討")
            
            # ESSに関する推奨
            low_ess = [name for name, ess in results['ess'].items() 
                      if not ess['adequate']]
            if low_ess:
                recommendations.append(f"- サンプリング効率の改善が必要: {', '.join(low_ess)}")
                recommendations.append("- 提案分布の調整または適応的MCMCの使用を検討")
            
            # 自己相関に関する推奨
            high_autocorr = [name for name, ac in results['autocorr'].items() 
                           if not ac['low_autocorr']]
            if high_autocorr:
                recommendations.append(f"- 自己相関の改善が必要: {', '.join(high_autocorr)}")
                recommendations.append("- 間引き (thinning) またはより効率的なサンプラーを検討")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        return recommendations
    
    def _save_results(self, results, filename):
        """結果をファイルに保存"""
        import json
        
        # NumPy配列をリストに変換
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        # 辞書を再帰的に変換
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        converted_results = convert_dict(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n診断結果を {filename} に保存しました")

# 包括的診断のデモ
def comprehensive_diagnostics_demo():
    """包括的診断のデモンストレーション"""
    
    # 第4章のベイズ線形回帰モデルを使用したシミュレーション
    np.random.seed(42)
    
    # 真のパラメータ
    true_beta = [2.0, -1.5, 0.8]
    true_sigma2 = 0.25
    true_params = true_beta + [true_sigma2]
    param_names = ['β₀', 'β₁', 'β₂', 'σ²']
    
    # 複数チェーンのシミュレーション
    n_chains = 3
    n_samples = 5000
    chains = []
    
    for chain_idx in range(n_chains):
        print(f"チェーン {chain_idx + 1} 生成中...")
        
        # 各チェーンで異なる初期値を使用
        initial_values = [[0, 0, 0], [1, -1, 1], [-1, 2, -0.5]]
        initial_sigma2s = [1.0, 0.5, 2.0]
        
        # 模擬的なMCMCサンプル生成
        chain_samples = []
        
        # ベータパラメータ
        for i, (true_val, init_val) in enumerate(zip(true_beta, initial_values[chain_idx])):
            samples = np.zeros(n_samples)
            samples[0] = init_val
            
            # 初期の収束期間
            for t in range(1, min(1000, n_samples)):
                drift = 0.02 * (true_val - samples[t-1])
                samples[t] = samples[t-1] + drift + np.random.normal(0, 0.05)
            
            # 定常期間
            for t in range(1000, n_samples):
                samples[t] = true_val + np.random.normal(0, 0.1)
                # 自己相関を追加
                if t > 1000:
                    correlation = 0.2 if chain_idx == 0 else 0.5 if chain_idx == 1 else 0.8
                    samples[t] = correlation * samples[t-1] + (1-correlation) * samples[t]
            
            chain_samples.append(samples)
        
        # シグマ²パラメータ
        sigma2_samples = np.random.gamma(10, true_sigma2/10, n_samples)
        # 初期値調整
        sigma2_samples[0] = initial_sigma2s[chain_idx]
        chain_samples.append(sigma2_samples)
        
        chains.append(np.column_stack(chain_samples))
    
    # 包括的診断の実行
    comprehensive_diagnostics = ComprehensiveMCMCDiagnostics(
        chains, param_names, true_params
    )
    
    results = comprehensive_diagnostics.run_all_diagnostics("mcmc_diagnostics_report.json")
    
    return comprehensive_diagnostics, results

# デモ実行
print("包括的MCMC診断のデモンストレーション...")
comprehensive_result, diag_results = comprehensive_diagnostics_demo()
```

この章では、MCMCサンプルの品質を評価し、収束を確認するための包括的な手法を解説しました。次章では、より高度なMCMC手法とその将来展望について学びます。

---