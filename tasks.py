"""Baitwatch's tasks for easy use."""

from invoke import task
import sys


# ----------------------------------
#        PACKAGE ACTIONS
# ----------------------------------

@task
def install_requirements(c):
    """Install requirements from requirements.txt."""
    c.run("pip install -r requirements.txt")


@task
def install(c):
    """Install the package in editable mode."""
    c.run("pip install -e . -U")


@task
def reinstall_package(c):
    """Reinstall the package (uninstall then install)."""
    c.run("pip uninstall -y baitwatch || :", hide=True)
    c.run("pip install -e .")


@task
def run_dl_data(c):
    """Download data for the project."""
    c.run("python -c 'from baitwatch.main import download_data; download_data()'")


@task
def run_preprocess_fonf(c):
    """Preprocess FONF data."""
    c.run("python -c 'from baitwatch.main import preprocess_data; preprocess_data(\"fonf\")'")


@task
def run_preprocess_ifsp(c):
    """Preprocess IFSP data."""
    c.run("python -c 'from baitwatch.main import preprocess_data; preprocess_data(\"ifsp\")'")


@task
def run_augment_ifsp(c):
    """Run data augmentation for IFSP."""
    c.run("python -c 'from baitwatch.main import save_augmented; save_augmented()'")


@task
def run_train_fonf(c):
    """Train model on FONF data."""
    c.run("python -c 'from baitwatch.main import train; train(\"fonf\")'")


@task
def run_train_ifsp(c):
    """Train model on IFSP data."""
    c.run("python -c 'from baitwatch.main import train; train(\"ifsp\")'")


@task
def run_train_ifsp_augmented(c):
    """Train model on IFSP augmented data."""
    c.run("python -c 'from baitwatch.main import train; train(\"ifsp\", augmented=True)'")


@task
def run_evaluate_fonf(c):
    """Evaluate model on FONF data."""
    c.run("python -c 'from baitwatch.main import evaluate; evaluate(\"fonf\")'")


@task
def run_evaluate_ifsp(c):
    """Evaluate model on IFSP data."""
    c.run("python -c 'from baitwatch.main import evaluate; evaluate(\"ifsp\")'")


@task
def run_cycle_fonf(c):
    """Run complete cycle for FONF data."""
    c.run("python -c 'from baitwatch.main import run_cycle; run_cycle(\"fonf\")'")


@task
def run_cycle_ifsp(c):
    """Run complete cycle for IFSP data."""
    c.run("python -c 'from baitwatch.main import run_cycle; run_cycle(\"ifsp\")'")


@task
def run_report_fonf(c):
    """Generate classification report for FONF data."""
    c.run("python -c 'from baitwatch.main import classification_report; classification_report(\"fonf\")'")


@task
def run_report_ifsp(c):
    """Generate classification report for IFSP data."""
    c.run("python -c 'from baitwatch.main import classification_report; classification_report(\"ifsp\")'")


@task
def run_api(c):
    """Run the FastAPI server with uvicorn."""
    c.run("uvicorn baitwatch.interfaces.api:app --reload")


# ----------------------------------
#        CONVENIENCE TASKS
# ----------------------------------

@task
def setup(c):
    """Complete setup: install requirements and package."""
    install_requirements(c)
    install(c)


@task
def preprocess(c, dataset="all"):
    """Preprocess data for specified dataset or all datasets."""
    if dataset == "all" or dataset == "fonf":
        run_preprocess_fonf(c)
    if dataset == "all" or dataset == "ifsp":
        run_preprocess_ifsp(c)


@task
def train(c, dataset="all", augmented=False):
    """Train model on specified dataset(s)."""
    if dataset == "all" or dataset == "fonf":
        run_train_fonf(c)
    if dataset == "all" or dataset == "ifsp":
        if augmented:
            run_train_ifsp_augmented(c)
        else:
            run_train_ifsp(c)


@task
def evaluate(c, dataset="all"):
    """Evaluate model on specified dataset(s)."""
    if dataset == "all" or dataset == "fonf":
        run_evaluate_fonf(c)
    if dataset == "all" or dataset == "ifsp":
        run_evaluate_ifsp(c)


@task
def cycle(c, dataset="all"):
    """Run complete cycle for specified dataset(s)."""
    if dataset == "all" or dataset == "fonf":
        run_cycle_fonf(c)
    if dataset == "all" or dataset == "ifsp":
        run_cycle_ifsp(c)


@task
def report(c, dataset="all"):
    """Generate classification report for specified dataset(s)."""
    if dataset == "all" or dataset == "fonf":
        run_report_fonf(c)
    if dataset == "all" or dataset == "ifsp":
        run_report_ifsp(c)


@task
def full_pipeline(c, dataset="all"):
    """Run complete pipeline: preprocess -> train -> evaluate -> report."""
    preprocess(c, dataset)
    train(c, dataset)
    evaluate(c, dataset)
    report(c, dataset)


