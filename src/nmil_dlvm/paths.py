from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"
DATASET_CONFIGS_ROOT = CONFIGS_ROOT / "datasets"
DATA_ROOT = REPO_ROOT / "data"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
SAVED_MODELS_ROOT = ARTIFACTS_ROOT / "models"
RESULTS_ROOT = ARTIFACTS_ROOT / "results"
VISUALIZATION_ROOT = ARTIFACTS_ROOT / "visualization"
MODEL_TRAINING_ANALYSIS_ROOT = ARTIFACTS_ROOT / "training"
ANALYSIS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "analysis"
PAPER_ROOT = REPO_ROOT / "paper"
MANUSCRIPT_FIGURES_ROOT = PAPER_ROOT / "figures"
PAPER_PDFS_ROOT = PAPER_ROOT / "pdfs"
DOCS_ROOT = REPO_ROOT / "docs"
OPS_ROOT = REPO_ROOT / "ops"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_config_path(dataset):
    return DATASET_CONFIGS_ROOT / f"{dataset}.py"


def data_dir(dataset):
    return DATA_ROOT / dataset


def model_training_analysis_dir(dataset):
    return MODEL_TRAINING_ANALYSIS_ROOT / dataset


def visualization_outputs_dir(dataset, model_id=None):
    path = VISUALIZATION_ROOT / "outputs" / dataset
    if model_id is not None:
        path = path / model_id
    return path


def visualization_presentations_dir(dataset):
    return VISUALIZATION_ROOT / "presentations" / dataset


def visualization_mle_dir(dataset):
    return VISUALIZATION_ROOT / "mle" / dataset


def results_dale_runs_dir():
    return RESULTS_ROOT / "dale_runs"


def ensure_dataset_runtime_dirs(dataset):
    ensure_dir(data_dir(dataset))
    ensure_dir(SAVED_MODELS_ROOT / dataset)
    ensure_dir(model_training_analysis_dir(dataset))
    ensure_dir(visualization_presentations_dir(dataset))
    ensure_dir(visualization_outputs_dir(dataset))
    ensure_dir(visualization_mle_dir(dataset))
