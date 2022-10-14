import os
import sys
import argparse
import json
import pandas as pd

import SQL_util


def create_parser():
    """
    Initialize argument parser
    """
    # Create parser
    parser = argparse.ArgumentParser()

    # Add city argument
    parser.add_argument("--model", help="model", required=True, action='store')
    parser.add_argument("--starttime", help="start stime", required=True, action='store')
    parser.add_argument("--endtime", help="end stime", required=True, action='store')
    parser.add_argument("--alpha", default=0.002, help="significance level", action='store')
    return parser


if __name__ == "__main__":

    # Are we on Linux of Windows? For Windows use py
    print("#---------------------------")
    print("# OS detected: %s" % os.name)
    print("#---------------------------")

    if os.name == 'posix':
        python_cmd = 'python'
    else:
        python_cmd = 'py'


    # Get project root
    root = os.path.join(os.getcwd()) 

    # Initialize parser
    parser = create_parser()

    # Parse command line argument
    args    = parser.parse_args()
    t_start = args.starttime
    print(t_start)
    model   = args.model
    t_end   = args.endtime
    alpha   = args.alpha

    # Database engine
    json_conf = os.path.join(root, "config_database.json")
    with open(json_conf, "r", encoding="utf-8")  as file:
            sql_config = json.load(file)
    engine = SQL_util.set_engine(sql_config['model_db'])

    # Fetch all valid flowmeter gids
    query = "select gid from flowmeter_config where gid < 1000"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    gids = [str(gid) for gid in df['gid'] if gid != 8]

    # Run main.py for each valid gid
    for gid in gids:
        cmd = "%s main.py --gid %s --model %s --starttime %s --endtime %s --alpha %s"% (
            python_cmd,
            gid, 
            model,
            f'"{t_start}"',
            f'"{t_end}"',
            alpha
        )
        print("#-----------------------------------")
        print("# GID = %s" % gid)
        print("#-----------------------------------")
        print(cmd)
        os.system(cmd)
        print("\n")


