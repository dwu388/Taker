from __future__ import annotations
import argparse,json
from .axiom_train_24h import train_models

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--output-dir',default='models/axiom24');ap.add_argument('--allow-small',action='store_true');a=ap.parse_args();print(json.dumps(train_models(a.db,a.output_dir,a.allow_small),indent=2,default=str))
if __name__=='__main__':main()
