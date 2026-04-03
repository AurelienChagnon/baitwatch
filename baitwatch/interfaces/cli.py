"""CLI interface."""

import argparse

from baitwatch.app import (
    augment,
    classification_report,
    download_data,
    evaluate,
    preprocess_data,
    run_cycle,
    train,
)


def main() -> None:
    """Main entry point for the baitwatch CLI."""
    parser = argparse.ArgumentParser(description="Baitwatch - Fish Detection Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download data command
    _ = subparsers.add_parser('download-data', help='Download data')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('dataset', choices=['fonf', 'ifsp'],
                                   help='Dataset to preprocess')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('dataset', choices=['fonf', 'ifsp'], help='Dataset to train on')
    train_parser.add_argument('--augmented', action='store_true',
                              help='Use augmented data')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('dataset', choices=['fonf', 'ifsp'], help='Dataset to evaluate')

    # Classification report command
    report_parser = subparsers.add_parser('report', help='Generate classification report')
    report_parser.add_argument('dataset', choices=['fonf', 'ifsp'],
                               help='Dataset to generate report for')
    report_parser.add_argument('--model-name', default='', help='Specific model name to use')

    # Run cycle command
    cycle_parser = subparsers.add_parser('cycle', help='Run complete cycle')
    cycle_parser.add_argument('dataset', choices=['fonf', 'ifsp'], help='Dataset to run cycle for')

    # Save augmented command
    augmented_parser = subparsers.add_parser('augment', help='Augment dataset')
    augmented_parser.add_argument('dataset', choices=['fonf', 'ifsp'], help='Dataset to augment')

    args = parser.parse_args()

    if args.command == 'download-data':
        download_data()
    elif args.command == 'preprocess':
        preprocess_data(args.dataset)
    elif args.command == 'train':
        train(args.dataset, augmented=args.augmented)
    elif args.command == 'evaluate':
        evaluate(args.dataset)
    elif args.command == 'report':
        classification_report(args.dataset, model_name=args.model_name)
    elif args.command == 'cycle':
        run_cycle(args.dataset)
    elif args.command == 'augment':
        augment(args.dataset)
    else:
        parser.print_help()
