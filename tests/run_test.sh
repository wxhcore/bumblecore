#!/bin/bash

pytest tests/bumblecore/data_processing/test_data_formatter.py -v -s
pytest tests/bumblecore/data_processing/test_datasets.py -v
pytest tests/bumblecore/cli/test_arg_parser.py -v
pytest tests/bumblecore/training/test_launcher.py -v
pytest tests/bumblecore/data_processing/test_packing.py -v