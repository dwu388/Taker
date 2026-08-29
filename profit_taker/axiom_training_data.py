from __future__ import annotations
import argparse
from .axiom_train_24h import load_training_frame

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--out',default='data/axiom_training_v18.csv');a=ap.parse_args();df=load_training_frame(a.db);df.to_csv(a.out,index=False);print({'rows':len(df),'out':a.out})
if __name__=='__main__':main()
