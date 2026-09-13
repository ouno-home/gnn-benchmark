import argparse
import gnnbench
import json
import os
import shlex

from gnnbench.util import get_pending_connection, fetch_pending_job, get_experiment_config
from pathlib import Path

# Get path to the `run_single_job.py` file which is called internally to execute the jobs.
SCRIPT_PATH = Path(gnnbench.__file__).parent / 'run_single_job.py'


def do_work(db_path, device, log_verbose):
    conn = get_pending_connection(db_path)

    job = fetch_pending_job(conn)
    if job is None:
        print("No pending jobs. Exiting...")
        return

    device_name = f"GPU {device}" if device is not None else "CPU"
    print(f"Running configs from database on {device_name}.\n\n")
    working_loop(conn, db_path, device=device, log_verbose=log_verbose)


def working_loop(conn, db_path, device, log_verbose):
    job = fetch_pending_job(conn)
    while job is not None:
        config = job["config"]
        config['db_path'] = db_path

        print(f"{config['experiment_name']}: Running split {config['split_no']} with seed {config['seed']}...\n---")
        config_string = f"{json.dumps(config)}"
        # escape bash metacharacters (mostly "")
        config_string = shlex.quote(config_string)

        command = f"python {SCRIPT_PATH} {device if device is not None else 'cpu'} {1 if log_verbose else 0} {config_string}"
        print(command)
        os.system(command)
        print(f"{config['experiment_name']}: Done split {config['split_no']} with seed {config['seed']}...")
        conn.execute("DELETE FROM pending WHERE id = ?", (job["id"],))
        conn.commit()

        # look if there's still work
        job = fetch_pending_job(conn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a worker process that fetches pending jobs from the '
                                                 '"pending" table of the SQLite database and executes them one by '
                                                 'one. Each worker runs on a single specified GPU device, or on CPU.')
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        help='Path to the YAML configuration file for the experiment.')
    parser.add_argument('-g',
                        '--gpu',
                        type=int,
                        required=False,
                        default=None,
                        help='The ID of the GPU to operate on. If not given, '
                             'the worker runs on CPU.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Display more log messages.')
    args = parser.parse_args()

    _experiment_config = get_experiment_config(args.config_file)
    _db_path = _experiment_config['db_path']
    do_work(_db_path, args.gpu, args.verbose)
