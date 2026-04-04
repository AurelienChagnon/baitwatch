"""CLI interface."""

import argparse

from baitwatch.app import (
    augment,
    classification_report,
    download_data,
    evaluate,
    preprocess_data,
    run_api,
    run_cycle,
    train,
)


def main() -> None:
    """Main entry point for the baitwatch CLI."""
    parser = argparse.ArgumentParser(description="Baitwatch - Fish Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download data command
    _ = subparsers.add_parser("download-data", help="Download data")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument(
        "dataset", choices=["fonf", "ifsp"], help="Dataset to preprocess"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("dataset", choices=["fonf", "ifsp"], help="Dataset to train on")
    train_parser.add_argument("--augmented", action="store_true", help="Use augmented data")

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    evaluate_parser.add_argument("dataset", choices=["fonf", "ifsp"], help="Dataset to evaluate")

    # Classification report command
    report_parser = subparsers.add_parser("report", help="Generate classification report")
    report_parser.add_argument(
        "dataset", choices=["fonf", "ifsp"], help="Dataset to generate report for"
    )
    report_parser.add_argument("--model-name", default="", help="Specific model name to use")

    # Run cycle command
    cycle_parser = subparsers.add_parser("cycle", help="Run complete cycle")
    cycle_parser.add_argument("dataset", choices=["fonf", "ifsp"], help="Dataset to run cycle for")

    # Save augmented command
    augmented_parser = subparsers.add_parser("augment", help="Augment dataset")
    augmented_parser.add_argument("dataset", choices=["fonf", "ifsp"], help="Dataset to augment")

    # API command
    api_parser = subparsers.add_parser("api", help="Run API")
    api_parser.add_argument("--host", default="127.0.0.1", help="Host to run API on")
    api_parser.add_argument("--port", default=8080, help="Port to run API on")
    api_parser.add_argument("--reload", action="store_true", help="Reload API on code changes")
    api_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    if args.command == "download-data":
        download_data()
    elif args.command == "preprocess":
        preprocess_data(args.dataset)
    elif args.command == "train":
        train(args.dataset, augmented=args.augmented)
    elif args.command == "evaluate":
        evaluate(args.dataset)
    elif args.command == "report":
        classification_report(args.dataset, model_name=args.model_name)
    elif args.command == "cycle":
        run_cycle(args.dataset)
    elif args.command == "augment":
        augment(args.dataset)
    elif args.command == "api":
        run_api(args.host, args.port, args.reload, args.workers)
    else:
        parser.print_help()
