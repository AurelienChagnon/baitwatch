"""Baitwatch's tasks for easy use."""

from invoke import Context, task


# ----------------------------------
#        PACKAGE ACTIONS
# ----------------------------------

@task
def install_requirements(context: Context) -> None:
    """Install requirements from requirements.txt."""
    context.run("pip install -r requirements.txt")


@task
def install(context: Context) -> None:
    """Install the package in editable mode."""
    context.run("pip install -e . -U")


@task
def reinstall_package(context: Context) -> None:
    """Reinstall the package (uninstall then install)."""
    context.run("pip uninstall -y baitwatch || :", hide=True)
    context.run("pip install -e .")


# ----------------------------------
#        CORE TASKS
# ----------------------------------

@task
def download_data(context: Context) -> None:
    """Download data for the project."""
    context.run("python -m baitwatch.main download-data")


@task
def preprocess(context: Context, dataset: str) -> None:
    """Preprocess data for specified dataset."""
    context.run(f"python -m baitwatch.main preprocess {dataset}")


@task
def train(context: Context, dataset: str, augmented: bool = False) -> None:
    """Train model on specified dataset."""
    augmented_flag = "--augmented" if augmented else ""
    context.run(f"python -m baitwatch.main train {dataset} {augmented_flag}")


@task
def evaluate(context: Context, dataset: str) -> None:
    """Evaluate model on specified dataset."""
    context.run(f"python -m baitwatch.main evaluate {dataset}")


@task
def report(context: Context, dataset: str, model_name: str = "") -> None:
    """Generate classification report for specified dataset."""
    model_flag = f"--model-name {model_name}" if model_name else ""
    context.run(f"python -m baitwatch.main report {dataset} {model_flag}")


@task
def cycle(context: Context, dataset: str) -> None:
    """Run complete cycle for specified dataset."""
    context.run(f"python -m baitwatch.main cycle {dataset}")


@task
def save_augmented(context: Context) -> None:
    """Save augmented IFSP dataset."""
    context.run("python -m baitwatch.main save-augmented")


@task
def api(context: Context, host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server."""
    reload_flag = "--reload" if reload else ""
    context.run(f"python -m baitwatch.interfaces.api --host {host} --port {port} {reload_flag}")


# ----------------------------------
#        CONVENIENCE TASKS
# ----------------------------------

@task
def setup(context: Context) -> None:
    """Complete setup: install requirements and package."""
    install_requirements(context)
    install(context)


@task
def full_pipeline(context: Context, dataset: str) -> None:
    """Run complete pipeline: preprocess -> train -> evaluate -> report."""
    preprocess(context, dataset)
    train(context, dataset)
    evaluate(context, dataset)
    report(context, dataset)


@task
def full_pipeline_augmented(context: Context, dataset: str) -> None:
    """Run complete pipeline with augmentation: preprocess -> train(augmented) -> evaluate -> report."""
    preprocess(context, dataset)
    train(context, dataset, augmented=True)
    evaluate(context, dataset)
    report(context, dataset)
