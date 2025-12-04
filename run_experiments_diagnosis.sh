#!/bin/bash

# --- 設定 ---
# 同時に実行する最大のジョブ数。マシンのCPUコア数などに合わせて調整してください。
MAX_JOBS=1

# スクリプトが置かれているディレクトリを基準にプロジェクトルートを設定
SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT="/home/sakoda/HDNNP_FORCE_PYTORCH/1001_kairyou_tuduki_tuduki_par"

# 結果を保存するベースディレクトリ
EXPERIMENTS_BASE_DIR="${PROJECT_ROOT}/experiments_output"


# --- 1つの実験を実行する関数 (変更あり) ---
run_experiment() {
    # 関数に渡された引数を読み込む
    RCUT=$1
    HIDDEN=$2
    ALPHA_E=$3
    BETA_F=$4

    # パラメータに基づいたディレクトリ名を生成
    HIDDEN_NAME=${HIDDEN// /_}
    RCUT_NAME="R${RCUT//./p}"
    ALPHA_E_NAME="A${ALPHA_E//./p}"
    BETA_F_NAME="B${BETA_F//./p}"
    RUN_NAME="${RCUT_NAME}_H${HIDDEN_NAME}_${ALPHA_E_NAME}_${BETA_F_NAME}"
    
    OUTPUT_DIR="${EXPERIMENTS_BASE_DIR}/${RUN_NAME}"

    # 各ジョブのログを個別のファイルにリダイレクト
    LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"
    mkdir -p "${OUTPUT_DIR}"

    # --- 実行ブロック ---
    # { ... } > で囲むことで、このブロック全体の標準出力と標準エラー出力をログファイルに送る
    {
        echo "========================================================================"
        echo "--- Starting Experiment: ${RUN_NAME}"
        echo "--- Output Directory:   ${OUTPUT_DIR}"
        echo "--- Log File:           ${LOG_FILE}"
        echo "--- Parameters:"
        echo "    R_CUT:          ${RCUT}"
        echo "    HIDDEN_LAYERS:  ${HIDDEN}"
        echo "    LOSS_ALPHA_E:   ${ALPHA_E}"
        echo "    LOSS_BETA_F:    ${BETA_F}"
        echo "========================================================================"
        
        # --- ステップ1: 学習の実行 ---
        echo ""
        echo "--- [Step 1/6] Running Training..."
        python "${PROJECT_ROOT}/scripts/run_train.py" \
            --R_CUT ${RCUT} \
            --HIDDEN_LAYERS ${HIDDEN} \
            --LOSS_ALPHA_E ${ALPHA_E} \
            --LOSS_BETA_F ${BETA_F} \
            --CHECKPOINT_DIR "${OUTPUT_DIR}"
        if [ $? -ne 0 ]; then echo "!!! Training failed for ${RUN_NAME}"; exit 1; fi
        echo "--- Training finished successfully."

        # --- ステップ2: 推論・評価の実行 ---
        echo ""
        echo "--- [Step 2/6] Running Inference..."
        python "${PROJECT_ROOT}/scripts/run_infer.py" \
            --CHECKPOINT_DIR "${OUTPUT_DIR}"
        if [ $? -ne 0 ]; then echo "!!! Inference failed for ${RUN_NAME}"; exit 1; fi
        echo "--- Inference finished successfully."
        
        # --- ステップ3: 記述子空間の分析 ---
        echo ""
        echo "--- [Step 3/6] Running Descriptor Analysis..."
        python "${PROJECT_ROOT}/scripts/analyze_descriptors.py" \
            --checkpoint-dir "${OUTPUT_DIR}"
        if [ $? -ne 0 ]; then echo "!!! Descriptor analysis failed for ${RUN_NAME}"; exit 1; fi
        echo "--- Descriptor analysis finished successfully."

        # --- ステップ4: 外れ値原子の特定 ---
        echo ""
        echo "--- [Step 4/6] Running Outlier Diagnosis..."
        python "${PROJECT_ROOT}/scripts/diagnose_outliers.py" \
            --checkpoint-dir "${OUTPUT_DIR}"
        if [ $? -ne 0 ]; then echo "!!! Outlier diagnosis failed for ${RUN_NAME}"; exit 1; fi
        echo "--- Outlier diagnosis finished successfully."

        # --- ステップ5: 局所環境の比較分析 ---
        echo ""
        echo "--- [Step 5/6] Running Environment Analysis..."
        python "${PROJECT_ROOT}/scripts/analyze_environment.py" \
            --checkpoint-dir "${OUTPUT_DIR}"
        if [ $? -ne 0 ]; then echo "!!! Environment analysis failed for ${RUN_NAME}"; exit 1; fi
        echo "--- Environment analysis finished successfully."

        # ▼▼▼ 追加されたステップ ▼▼▼
        # --- ステップ6: 外挿の可視化分析 ---
        echo ""
        echo "--- [Step 6/6] Running Extrapolation Analysis..."
        python "${PROJECT_ROOT}/scripts/analyze_extrapolation.py" \
            --checkpoint-dir "${OUTPUT_DIR}"
        if [ $? -ne 0 ]; then echo "!!! Extrapolation analysis failed for ${RUN_NAME}"; exit 1; fi
        echo "--- Extrapolation analysis finished successfully."
        # ▲▲▲ ここまで ▲▲▲

        echo "--- Experiment ${RUN_NAME} completed."

    } > "${LOG_FILE}" 2>&1
    # --------------------
}


# --- ここから実行したい実験のパラメータを配列として定義 ---
declare -a experiments
experiments=(
    "5.0 '20 20' 1.0 0.1"
)


# --- 並列実行のメインロジック ---
echo "Starting experiment suite. Max parallel jobs: ${MAX_JOBS}"
for params in "${experiments[@]}"; do
    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi
    eval "run_experiment ${params}" &
done

wait

echo ""
echo "========================================================================"
echo "--- All experiments finished successfully. ---"
echo "========================================================================"