# HDNNP Project README

本プロジェクトは、PyTorchを用いた高次元ニューラルネットワークポテンシャル (High-Dimensional Neural Network Potential: HDNNP) の実装です。VASPの計算結果からデータを抽出し、モデルの学習、推論、および詳細な分析（記述子空間、外挿、局所環境など）を行うためのスクリプト群が含まれています。

## 1. ディレクトリ構造

プロジェクトの主要なディレクトリとファイルの配置は以下の通りです。

```text
project_root/
├── data/
│   ├── raw/                 # VASPの生データ (XDATCAR, OUTCAR, OSZICAR) を配置
│   └── processed/           # 抽出・変換されたデータ (.npz) と splits.json が保存される
├── experiments_output/      # 実験ごとのログ、モデル、プロット結果の出力先
├── hdnnp/                   # HDNNPのコアパッケージ (ライブラリ)
│   ├── __init__.py          # パッケージ初期化
│   ├── config.py            # 全体の設定ファイル (パス、ハイパーパラメータ、物理定数)
│   ├── data_loader.py       # データ読み込み、バッチ化処理 (Dataset, DataLoader)
│   ├── loss.py              # 損失関数の定義 (エネルギーRMSE + 力RMSE)
│   ├── model.py             # ニューラルネットワークモデル定義 (HDNNPModel, ElementNN)
│   ├── pytorch_ver.py       # 依存ライブラリのチェック用ユーティリティ
│   ├── symmetry_calculator.py # 対称性関数 (G2, G3) の計算クラス (近傍探索含む)
│   ├── symmetry_functions.py  # 対称性関数の数式定義 (低レイヤー)
│   └── train.py             # 学習ループのロジック (Optimizer, EarlyStopping)
├── scripts/                 # 実行用スクリプト群
│   ├── run_extract.py       # 前処理: VASPデータ抽出 -> .npz作成
│   ├── make_splits.py       # 前処理: データセット分割 -> splits.json作成
│   ├── run_train.py         # 学習実行
│   ├── run_infer.py         # 推論・評価実行
│   ├── analyze_descriptors.py   # 分析: 記述子空間のPCA分析
│   ├── analyze_extrapolation.py # 分析: 外挿検知のためのPES可視化
│   ├── analyze_environment.py   # 分析: 外れ値原子の局所環境分析
│   └── diagnose_outliers.py     # 分析: PCAに基づく外れ値原子の特定
└── run_experiments_diagnosis.sh # 実験自動化用シェルスクリプト
```
## 2. 実行フロー (計算の流れ)

基本的な計算の流れは以下の順序で行います。

### Step 0: 設定
`hdnnp/config.py` を編集し、ディレクトリパス、対象元素 (SPECIES)、カットオフ半径 (R_CUT)、対称性関数のパラメータなどを設定します。

### Step 1: データの前処理 (抽出)
VASPの計算結果から学習用データを抽出します。

* **実行ファイル**: `scripts/run_extract.py`
* **入力**: `data/raw/` 以下のVASP計算ディレクトリ (XDATCAR, OUTCAR, OSZICAR)
* **出力**: `data/processed/` 以下の `.npz` ファイル (1フレーム1ファイル)
* **コマンド例**:
    ```bash
    python scripts/run_extract.py
    ```

### Step 2: データセットの分割
抽出したデータを学習(train)・検証(valid)・テスト(test)に分割します。

* **実行ファイル**: `scripts/make_splits.py`
* **入力**: `data/processed/` 内の `.npz` ファイル群
* **出力**: `data/processed/splits.json`
* **コマンド例**:
    ```bash
    python scripts/make_splits.py --train-fraction 0.8 --valid-fraction 0.1
    ```

### Step 3: 実験の実行 (学習〜分析)
ここからは個別のスクリプトを実行することも可能ですが、自動化スクリプトを使用するとパラメータを変えた実験を一括管理できます。

#### A. 自動化スクリプトを使用する場合 (推奨)
学習、推論、そして一連の分析スクリプトを順番に実行します。

* **実行ファイル**: `run_experiments_diagnosis.sh`
* **内容**: `run_train.py` -> `run_infer.py` -> `analyze_descriptors.py` -> `diagnose_outliers.py` -> `analyze_environment.py` -> `analyze_extrapolation.py` の順に実行。
* **コマンド例**:
    ```bash
    bash run_experiments_diagnosis.sh
    ```

#### B. 個別に実行する場合

* **学習 (`scripts/run_train.py`)**
    モデルを学習し、チェックポイント (`model_fullbatch.pt`) を保存します。
    ```bash
    python scripts/run_train.py --CHECKPOINT_DIR experiments_output/test_run
    ```

* **推論 (`scripts/run_infer.py`)**
    学習済みモデルで推論を行い、エネルギー・力の精度評価とプロット作成を行います。
    ```bash
    python scripts/run_infer.py --CHECKPOINT_DIR experiments_output/test_run
    ```

* **各種分析**
    必要に応じて `scripts/analyze_*.py` や `scripts/diagnose_outliers.py` を実行します。

---

## 3. 各ファイルの役割詳細

### コアライブラリ (`hdnnp/`)
スクリプトから呼び出される共通処理が記述されています。

* **`config.py`**:
    プロジェクトの「司令塔」です。全てのパス、モデル構造（中間層のサイズ）、対称性関数のパラメータ、物理定数、学習設定（Epoch数、Optimizer）を一元管理します。
* **`symmetry_calculator.py`**:
    原子座標(R)とセル情報から、記述子となる対称性関数(G)を計算します。PyTorchを用いてGPU上での高速計算を行い、最小イメージ規則(MIC)による近傍探索も実装されています。
* **`model.py`**:
    PyTorchの `nn.Module` を継承したモデル定義です。
    * `HDNNPModel`: 全体のモデル。元素ごとのネットワークを束ねます。
    * `ElementNN`: 元素ごとのサブネットワーク (Feed Forward Network)。
* **`train.py`**:
    学習ループの実装です。LBFGSやAdamなどのオプティマイザ制御、Early Stopping、ログ出力を行います。
* **`data_loader.py`**:
    大量の `.npz` ファイルを効率的に読み込むための Dataset クラスと、原子数が異なる構造をバッチ処理するためのパディング処理 (`pad_collate_fn`) を提供します。
* **`loss.py`**:
    エネルギーの誤差と力の誤差を重み付けして合算する損失関数を定義しています。

### 実行スクリプト (`scripts/`)
ユーザーが直接コマンドラインから実行するファイル群です。

* **`run_extract.py`**:
    VASPの出力ファイルを解析し、HDNNP学習に必要な形式 (.npz) に変換します。エネルギー、力、座標、セル情報を抽出します。
* **`make_splits.py`**:
    .npz ファイルリストをシャッフルし、指定された割合で train/valid/test に分割してJSONファイルに保存します。
* **`run_train.py`**:
    学習のエントリーポイントです。Config設定やCLI引数に基づき、データローダーとモデルを初期化し、学習を開始します。結果は指定されたチェックポイントディレクトリに保存されます。
* **`run_infer.py`**:
    学習済みモデルをロードし、テストデータ（または学習データ）に対する予測精度を評価します。実測値と予測値の相関プロットなどを生成します。
* **`analyze_descriptors.py`**:
    学習データとテストデータの記述子（対称性関数の値）を計算し、PCA（主成分分析）を用いて2次元に圧縮して可視化します。データの分布に偏りがないか確認するために使用します。
* **`diagnose_outliers.py`**:
    記述子空間のPCA結果に基づき、テストデータ内で分布から外れた「外れ値原子」を特定し、CSVに出力します。
* **`analyze_environment.py`**:
    `diagnose_outliers.py` で特定された外れ値原子について、その局所環境（配位数や近接原子との距離）を詳細に分析し、レポートを出力します。学習データの不足箇所を特定するのに役立ちます。
* **`analyze_extrapolation.py`**:
    記述子の各成分について、モデルが学習したポテンシャルエネルギー曲面(PES)の断面を可視化します。学習データの範囲外（外挿領域）でモデルが不自然な挙動をしていないか確認します。

### 自動化スクリプト

* **`run_experiments_diagnosis.sh`**:
    パラメータ（カットオフ半径や隠れ層サイズなど）を指定して、学習から詳細分析までを一気通貫で実行するためのBashスクリプトです。複数の条件で実験を回す際に便利です。