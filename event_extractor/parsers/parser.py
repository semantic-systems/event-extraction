import argparse


def parse():
    parser = argparse.ArgumentParser(description=' and the job name.')
    parser.add_argument('config', metavar='C', type=str,
                        help='Path to a configuration file (.yaml)')
    parser.add_argument('job_name', metavar='j', type=str,
                        help='Name of this job as an ID.')
    parser.parse_args()
    return parser.parse_args()