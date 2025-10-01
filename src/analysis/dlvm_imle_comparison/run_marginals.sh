#!/bin/bash

# ==============================================================================
# 运行 compare_imle_params.py 脚本以处理不同的N值
# ==============================================================================

# 设置脚本在遇到错误时立即退出，这是一种良好的编程习惯
set -e

# --- 用户配置 ---

# 1. Python脚本的路径
#    假设此脚本与您的Python脚本在同一目录下
PYTHON_SCRIPT="plot_marginal_median.py"

# 2. 参数文件夹的名称
#    这是传递给 --params_dir 的值
PARAMS_DIR="5_sim_1_restarts_param_fits"

# 3. Ground Truth参数文件名
#    **请注意**: 根据您的描述“使用的ground_truth所在文件夹是5_sim_data”，
#    我们假设在 compare_imle_params.py 内部引用的 `DATASET` 变量值就是 '5_sim_data'。
#    因此，这里我们只需要提供该文件夹下的具体文件名。
#    请根据您的实际情况修改下面的文件名。

GROUND_TRUTH_FILE="all_data-best_mle_params_mpf100.pt"

# 定义要运行的N值列表
N_VALUES=(2 50)

echo "自动化运行脚本已启动..."
echo "使用的参数文件夹: ${PARAMS_DIR}"
echo "使用的Ground Truth文件: ${GROUND_TRUTH_FILE}"
echo "----------------------------------------------------"


# 遍历N值列表并执行Python脚本
for N in "${N_VALUES[@]}"; do
  echo "[*] 正在为 N=${N} 运行..."

#  python3 "${PYTHON_SCRIPT}" \
#    --N "${N}" \
#    --params_dir "${PARAMS_DIR}" \
#    --ground_truth_pt_file "${GROUND_TRUTH_FILE}"

  # with grid_search
   python3 "${PYTHON_SCRIPT}" \
     --N "${N}" \
     --params_dir "${PARAMS_DIR}" \
     --ground_truth_pt_file "${GROUND_TRUTH_FILE}" \
     --plot_grid_search \
     --data_path "5_sim_data"

  echo "[✓] N=${N} 的任务已完成。"
  echo "----------------------------------------------------"
done

echo "所有任务已成功完成！生成的图片位于 output 目录中。"