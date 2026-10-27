import argparse,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
import uwb_search_v4 as engine
p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--split',choices=['val','test'],default='val')
p.add_argument('--cache',type=Path)
a=p.parse_args()
engine.ROOT=Path(__file__).resolve().parent/'results'
sys.path.insert(0,str(a.data/'src'))
import abc2027.evaluate as ev
with threadpool_limits(limits=4):engine.predict(a,ev)
