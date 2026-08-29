from __future__ import annotations
import argparse,json
from .config import load_env

def masked(v): return None if not v else (v[:4]+'...'+v[-4:] if len(v)>10 else 'configured')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-file',default='.env');a=ap.parse_args();e=load_env(a.env_file)
    print(json.dumps({k.replace('_API_KEY','').title():('configured' if e.get(k) else 'missing') for k in ('HELIUS_API_KEY','SHYFT_API_KEY','BIRDEYE_API_KEY')},indent=2))
if __name__=='__main__':main()
