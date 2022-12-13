import logging
import sys

stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format="%(asctime)s|%(name)-25s|%(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s")
for name in ['boto', 'urllib3', 's3transfer', 'boto3', 'botocore', 'nose']:
    logging.getLogger(name).setLevel(logging.INFO)


def get_logger(name: str):
    return logging.getLogger(name)
