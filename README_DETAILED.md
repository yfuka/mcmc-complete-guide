# MCMC（マルコフ連鎖モンテカルロ法）完全ガイド

MCMCに関する諸学者向けの教育コンテンツです。6つのJupyter Notebookで構成されており、基礎理論から高度な手法まで体系的に学習できます。

## 内容構成

### Chapter 1: MCMC基礎理論 (`chapter1_mcmc_basics.ipynb`)
- MCMCの基本概念と必要性
- マルコフ連鎖の基本性質
- 定常分布と詳細釣り合い条件

### Chapter 2: メトロポリス・ヘイスティングス法 (`chapter2_metropolis_hastings.ipynb`)
- アルゴリズムの詳細実装
- 提案分布の選択と性能への影響
- 多変量分布への拡張

### Chapter 3: ギブスサンプリング (`chapter3_gibbs_sampling.ipynb`)
- 条件付き分布による逐次サンプリング
- 混合モデルの推定
- ブロックギブスサンプリング

### Chapter 4: 収束診断と性能評価 (`chapter4_convergence_diagnostics.ipynb`)
- 視覚的・数値的診断手法
- Gelman-Rubin統計量
- 実践的な診断ワークフロー

### Chapter 5: 実践的応用例 (`chapter5_practical_applications.ipynb`)
- ベイズ線形回帰・ロジスティック回帰
- 階層モデルの実装
- 実データを用いた分析例

### Chapter 6: 高度なMCMC手法 (`chapter6_advanced_mcmc.ipynb`)
- ハミルトニアンモンテカルロ法（HMC）
- No-U-Turn Sampler（NUTS）
- 実用的なライブラリの紹介

## 環境構築

### uvを使用する場合（推奨）

```bash
# プロジェクトのクローン/ダウンロード後
cd mcmc

# 依存関係のインストール
uv sync

# Jupyter Notebookの起動
uv run jupyter notebook
```

### pipを使用する場合

```bash
# 仮想環境の作成と有効化
python -m venv mcmc_env
source mcmc_env/bin/activate  # Windowsの場合: mcmc_env\Scripts\activate

# 依存関係のインストール
pip install numpy matplotlib seaborn scipy pandas scikit-learn statsmodels jupyter ipykernel plotly tqdm

# Jupyter Notebookの起動
jupyter notebook
```

## 必要なライブラリ

- **numpy**: 数値計算の基盤
- **matplotlib**: 基本的な可視化
- **seaborn**: 統計的可視化
- **scipy**: 科学計算（統計分布、最適化等）
- **pandas**: データ処理・分析
- **scikit-learn**: 機械学習（データ生成、評価指標等）
- **statsmodels**: 統計モデリング（自己相関関数等）
- **jupyter**: Jupyter Notebook環境
- **plotly**: インタラクティブな可視化
- **tqdm**: プログレスバー

## 使用方法

1. 環境構築を完了させる
2. Jupyter Notebookを起動
3. Chapter 1から順番に実行・学習
4. 各章の演習問題に取り組む
5. 実際のデータで応用を試す

## 学習目標

このコンテンツを完了すると、以下ができるようになります：

- MCMCの理論的基礎の理解
- 各種MCMC手法の実装
- 収束診断と性能評価
- 実際の統計問題への応用
- 適切な手法選択と実用的な運用

## 対象者

- 統計学・機械学習を学ぶ学生・研究者
- ベイズ統計に興味のあるデータサイエンティスト
- MCMCの理論と実装を体系的に学びたい方

## ライセンス

教育目的での使用を想定しています。内容の改変・再配布は自由ですが、出典を明記してください。