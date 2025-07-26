# MCMC（マルコフ連鎖モンテカルロ法）完全ガイド

## 1. はじめに

MCMC（Markov Chain Monte Carlo）は、複雑な確率分布からサンプリングを行うための強力な手法です。直接サンプリングが困難な高次元の確率分布から、マルコフ連鎖を用いて間接的にサンプルを生成します。

### MCMCが必要な理由
- 正規化定数の計算が困難な分布からのサンプリング
- 高次元空間での積分計算
- ベイズ推論における事後分布からのサンプリング

## 2. 基礎理論

### 2.1 マルコフ連鎖の基本

マルコフ連鎖とは、現在の状態のみに依存して次の状態が決まる確率過程です。

**マルコフ性**：
$$P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t)$$

### 2.2 定常分布

マルコフ連鎖が十分長い時間経過後に収束する分布を定常分布と呼びます。

**定常分布の条件**：
$$\pi = \pi P$$

ここで、$\pi$は定常分布、$P$は遷移確率行列です。

### 2.3 詳細釣り合い条件

MCMCアルゴリズムの多くは詳細釣り合い条件を満たします：

$$\pi(x) P(x \rightarrow y) = \pi(y) P(y \rightarrow x)$$

## 3. メトロポリス・ヘイスティングス法

### 3.1 アルゴリズム

1. 現在の状態を$x$とする
2. 提案分布$q(x'|x)$から新しい状態$x'$を提案
3. 受理確率を計算：
   $$\alpha = \min\left(1, \frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}\right)$$
4. 確率$\alpha$で$x'$を受理、そうでなければ$x$に留まる

### 3.2 Python実装例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def metropolis_hastings(target_pdf, proposal_sampler, initial_value, n_samples):
    """
    メトロポリス・ヘイスティングス法の実装
    
    Parameters:
    - target_pdf: 目標分布の確率密度関数（正規化不要）
    - proposal_sampler: 提案分布からのサンプラー関数
    - initial_value: 初期値
    - n_samples: サンプル数
    """
    samples = np.zeros(n_samples)
    current = initial_value
    n_accepted = 0
    
    for i in range(n_samples):
        # 新しい状態を提案
        proposed = proposal_sampler(current)
        
        # 受理確率を計算
        alpha = min(1, target_pdf(proposed) / target_pdf(current))
        
        # 受理/棄却を決定
        if np.random.rand() < alpha:
            current = proposed
            n_accepted += 1
        
        samples[i] = current
    
    acceptance_rate = n_accepted / n_samples
    return samples, acceptance_rate

# 例：混合正規分布からのサンプリング
def target_pdf(x):
    """2つの正規分布の混合"""
    return 0.3 * stats.norm.pdf(x, -2, 0.5) + 0.7 * stats.norm.pdf(x, 2, 1)

def proposal_sampler(x):
    """正規分布による提案"""
    return x + np.random.normal(0, 0.5)

# サンプリング実行
samples, acceptance_rate = metropolis_hastings(
    target_pdf, proposal_sampler, 0, 10000
)

print(f"受理率: {acceptance_rate:.2%}")

# 結果の可視化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(samples[:1000])
plt.title('マルコフ連鎖のトレース')
plt.xlabel('ステップ')
plt.ylabel('値')

plt.subplot(1, 3, 2)
plt.hist(samples[1000:], bins=50, density=True, alpha=0.7, label='MCMC samples')
x = np.linspace(-5, 5, 1000)
plt.plot(x, target_pdf(x), 'r-', label='Target distribution')
plt.legend()
plt.title('サンプルの分布')

plt.subplot(1, 3, 3)
plt.acf(samples[1000:], lags=50)
plt.title('自己相関関数')
plt.tight_layout()
plt.show()
```

## 4. ギブスサンプリング

### 4.1 理論

ギブスサンプリングは、多変量分布から各変数を条件付き分布に基づいて順番に更新する手法です。

**アルゴリズム**（2変数の場合）：
1. $x^{(t+1)} \sim p(x | y^{(t)})$
2. $y^{(t+1)} \sim p(y | x^{(t+1)})$

### 4.2 実装例：2変量正規分布

```python
def gibbs_sampling_bivariate_normal(mu, cov, n_samples):
    """
    2変量正規分布からのギブスサンプリング
    
    Parameters:
    - mu: 平均ベクトル [mu_x, mu_y]
    - cov: 共分散行列
    - n_samples: サンプル数
    """
    samples = np.zeros((n_samples, 2))
    
    # 条件付き分布のパラメータを計算
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[1, 1])
    
    # 初期値
    x, y = 0, 0
    
    for i in range(n_samples):
        # x | y からサンプリング
        mu_x_given_y = mu[0] + rho * sigma_x / sigma_y * (y - mu[1])
        sigma_x_given_y = sigma_x * np.sqrt(1 - rho**2)
        x = np.random.normal(mu_x_given_y, sigma_x_given_y)
        
        # y | x からサンプリング
        mu_y_given_x = mu[1] + rho * sigma_y / sigma_x * (x - mu[0])
        sigma_y_given_x = sigma_y * np.sqrt(1 - rho**2)
        y = np.random.normal(mu_y_given_x, sigma_y_given_x)
        
        samples[i] = [x, y]
    
    return samples

# パラメータ設定
mu = np.array([1, 2])
cov = np.array([[1, 0.8], [0.8, 2]])

# サンプリング
samples = gibbs_sampling_bivariate_normal(mu, cov, 5000)

# 可視化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(samples[:500, 0], samples[:500, 1], 'b-', alpha=0.5, linewidth=0.5)
plt.plot(samples[:500, 0], samples[:500, 1], 'b.', markersize=2)
plt.title('ギブスサンプリングの軌跡')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.scatter(samples[1000:, 0], samples[1000:, 1], alpha=0.5, s=1)
plt.title('サンプルの散布図')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()
```

## 5. 収束診断

### 5.1 視覚的診断

```python
def plot_mcmc_diagnostics(samples, param_name="parameter"):
    """MCMCの収束診断プロット"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # トレースプロット
    axes[0, 0].plot(samples)
    axes[0, 0].set_title('トレースプロット')
    axes[0, 0].set_xlabel('イテレーション')
    axes[0, 0].set_ylabel(param_name)
    
    # ヒストグラム
    axes[0, 1].hist(samples, bins=50, density=True)
    axes[0, 1].set_title('事後分布')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('密度')
    
    # 自己相関
    from statsmodels.tsa.stattools import acf
    lags = min(100, len(samples) // 4)
    autocorr = acf(samples, nlags=lags)
    axes[1, 0].plot(autocorr)
    axes[1, 0].axhline(0, color='k', linestyle='--')
    axes[1, 0].set_title('自己相関関数')
    axes[1, 0].set_xlabel('ラグ')
    axes[1, 0].set_ylabel('ACF')
    
    # 累積平均
    cumulative_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    axes[1, 1].plot(cumulative_mean)
    axes[1, 1].set_title('累積平均')
    axes[1, 1].set_xlabel('イテレーション')
    axes[1, 1].set_ylabel('平均')
    
    plt.tight_layout()
    plt.show()
```

### 5.2 Gelman-Rubinの$\hat{R}$統計量

```python
def gelman_rubin_diagnostic(chains):
    """
    Gelman-Rubin収束診断統計量の計算
    
    Parameters:
    - chains: shape (n_chains, n_samples) の配列
    """
    n_chains, n_samples = chains.shape
    
    # 各チェーンの平均と分散
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    
    # チェーン間分散
    B = n_samples * np.var(chain_means, ddof=1)
    
    # チェーン内分散
    W = np.mean(chain_vars)
    
    # 事後分散の推定値
    var_plus = ((n_samples - 1) * W + B) / n_samples
    
    # R-hat統計量
    R_hat = np.sqrt(var_plus / W)
    
    return R_hat
```

## 6. 実践例：ベイズ線形回帰

```python
def bayesian_linear_regression_mcmc(X, y, n_samples=10000):
    """
    ベイズ線形回帰のMCMC実装
    """
    n, p = X.shape
    
    # 事前分布のパラメータ
    beta_prior_mean = np.zeros(p)
    beta_prior_precision = 0.01 * np.eye(p)
    alpha_sigma = 1.0
    beta_sigma = 1.0
    
    # 初期値
    beta = np.zeros(p)
    sigma2 = 1.0
    
    # サンプル保存用
    beta_samples = np.zeros((n_samples, p))
    sigma2_samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        # beta | sigma2, y のサンプリング
        precision = X.T @ X / sigma2 + beta_prior_precision
        precision_inv = np.linalg.inv(precision)
        mean = precision_inv @ (X.T @ y / sigma2 + beta_prior_precision @ beta_prior_mean)
        beta = np.random.multivariate_normal(mean, precision_inv)
        
        # sigma2 | beta, y のサンプリング
        residuals = y - X @ beta
        shape = alpha_sigma + n / 2
        scale = beta_sigma + np.sum(residuals**2) / 2
        sigma2 = 1 / np.random.gamma(shape, 1/scale)
        
        beta_samples[i] = beta
        sigma2_samples[i] = sigma2
    
    return beta_samples, sigma2_samples

# データ生成
np.random.seed(42)
n = 100
X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
true_beta = np.array([1, 2, -1])
y = X @ true_beta + np.random.normal(0, 0.5, n)

# MCMC実行
beta_samples, sigma2_samples = bayesian_linear_regression_mcmc(X, y)

# 結果の表示
print("パラメータの事後平均:")
print(f"beta_0: {np.mean(beta_samples[2000:, 0]):.3f} (真値: {true_beta[0]})")
print(f"beta_1: {np.mean(beta_samples[2000:, 1]):.3f} (真値: {true_beta[1]})")
print(f"beta_2: {np.mean(beta_samples[2000:, 2]):.3f} (真値: {true_beta[2]})")
print(f"sigma^2: {np.mean(sigma2_samples[2000:]):.3f} (真値: 0.25)")
```

## 7. 高度なトピック

### 7.1 ハミルトニアンモンテカルロ法（HMC）

HMCは物理学の概念を用いて効率的なサンプリングを実現します：

```python
def simple_hmc(log_prob, grad_log_prob, initial_value, n_samples, epsilon=0.01, L=10):
    """
    簡易版ハミルトニアンモンテカルロ法
    """
    samples = np.zeros((n_samples, len(initial_value)))
    current_q = initial_value
    
    for i in range(n_samples):
        q = current_q
        p = np.random.normal(size=len(q))  # 運動量のサンプリング
        current_p = p
        
        # リープフロッグ積分
        p = p + epsilon * grad_log_prob(q) / 2
        for _ in range(L):
            q = q + epsilon * p
            if _ != L - 1:
                p = p + epsilon * grad_log_prob(q)
        p = p + epsilon * grad_log_prob(q) / 2
        p = -p
        
        # メトロポリス補正
        current_H = -log_prob(current_q) + 0.5 * np.sum(current_p**2)
        proposed_H = -log_prob(q) + 0.5 * np.sum(p**2)
        
        if np.random.rand() < np.exp(current_H - proposed_H):
            current_q = q
        
        samples[i] = current_q
    
    return samples
```

### 7.2 適応的MCMC

提案分布を自動的に調整する手法：

```python
def adaptive_metropolis(target_pdf, initial_value, n_samples, adaptation_interval=100):
    """
    適応的メトロポリス法
    """
    dim = len(initial_value)
    samples = np.zeros((n_samples, dim))
    current = initial_value
    
    # 初期共分散行列
    cov = 0.1 * np.eye(dim)
    mean = initial_value.copy()
    
    for i in range(n_samples):
        # 提案
        proposed = np.random.multivariate_normal(current, cov)
        
        # 受理確率
        alpha = min(1, target_pdf(proposed) / target_pdf(current))
        
        if np.random.rand() < alpha:
            current = proposed
        
        samples[i] = current
        
        # 共分散行列の適応的更新
        if i > 0 and i % adaptation_interval == 0:
            mean = np.mean(samples[:i], axis=0)
            cov = np.cov(samples[:i].T) + 1e-6 * np.eye(dim)
    
    return samples
```

## 8. まとめと実践的アドバイス

### MCMCを使う際のベストプラクティス

1. **複数のチェーンを実行**：収束診断のため、異なる初期値から複数のチェーンを実行
2. **バーンイン期間の設定**：初期のサンプルは破棄
3. **thinning（間引き）**：自己相関を減らすため、k個ごとにサンプルを保存
4. **収束診断の実施**：視覚的診断と数値的診断の両方を使用
5. **効率的なアルゴリズムの選択**：問題に応じてMH法、ギブス、HMCなどを選択

### 参考文献
- Robert, C. & Casella, G. (2004). Monte Carlo Statistical Methods
- Brooks, S. et al. (2011). Handbook of Markov Chain Monte Carlo
- Gelman, A. et al. (2013). Bayesian Data Analysis