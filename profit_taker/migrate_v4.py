from __future__ import annotations
import argparse,json
from .db import migrate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');a=ap.parse_args();migrate(a.db);print(json.dumps({'migrated':a.db,'schema':'rebuilt-v18'}))
if __name__=='__main__':main()
