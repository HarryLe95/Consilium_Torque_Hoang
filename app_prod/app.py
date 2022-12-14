# -*- coding: utf-8 -*-
"""
Created: 2022-09-09
@author: Steve Lechowicz

Santos Advanced Analytics Sandbox
Standard application entry point for SandboxProductionLoop (SPL) architecture

NOTE: Santos standards mandate NO HARD-CODING of credentials in any source code.
"""
import argparse
import config.__config__ as base_config
import config.__state__ as base_state
import os
import random
import utils.advancedanalytics_util as aau
from utils.logging import get_logger
from datetime import datetime
from Model.Trainer import Trainer

logger = get_logger(__name__)

class ArgParser:
    def __init__(self, description):
        self._description = description

    def run_arg_parser(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=self._description)
        parser_with_args = self._add_parser_args(parser)
        args = parser_with_args.parse_args()
        return args

    @staticmethod
    def _add_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--group_id", nargs='?', default=0, help="Integer group_id for parallelisation", type=int)
        return parser


def main():
    # Base configuration and state
    config = base_config.init()
    state = base_state.init()

    # Environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["SM_FRAMEWORK"] = "tf.keras"

    # play around without the seed value
    random.seed(22)

    # Command line arguments
    arg_parser = ArgParser(description=config['application_name'])
    args = arg_parser.run_arg_parser()

    logger.info("Instantiating connections")

    # Group ids can be matched against instance hostnames to provide simple parallelisation
    if args.group_id > 0:
        groups = [args.group_id]
    else:
        groups = [0]
    state['alldatauptodate'] = False
    runstart = datetime.utcnow()
    while (True):
        totalruntime = (datetime.utcnow() - runstart).total_seconds()
        if state['alldatauptodate'] or totalruntime >= config['maxruntime_seconds']:
            break
        state['alldatauptodate'] = True
        for group_id in groups:
            state['group_id'] = group_id
            roc_model = Trainer(config['group_info'],
                            config['inference_info'],
                            config['data_connection_info'],
                            config['torque_info'],
                            config['model_info'])
            if config['perform_model_training']:
                logger.info("Model training start")
                roc_model.run_model_training()
                logger.info("Model training complete")
            if config['perform_model_inference']:
                logger.info("Model inference start")
                roc_model.run_model_inference()
                logger.info("Model inference complete")

if __name__ == "__main__":
    main()
