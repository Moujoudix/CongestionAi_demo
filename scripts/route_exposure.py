from __future__ import annotations
import pandas as pd, numpy as np, json, math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def exposure_for_departure(preds_csv: str, route_json: str, depart_time: str, speed_kmh: float = 40.0):
    df = pd.read_csv(preds_csv, parse_dates=['time'])
    with open(route_json,'r') as f: coords = json.load(f)
    samples, segm = [], []
    for (x1,y1),(x2,y2) in zip(coords[:-1], coords[1:]):
        m = haversine(y1,x1,y2,x2); n = max(1, int(m/500))
        for i in range(n):
            t = i/n; xs = x1 + t*(x2-x1); ys = y1 + t*(y2-y1)
            samples.append((ys,xs)); segm.append(m/n)
    samples.append((coords[-1][1], coords[-1][0])); segm.append(0)
    speed_ms = speed_kmh*1000/3600.0
    timesec = np.cumsum([s/speed_ms for s in segm])
    depart = pd.to_datetime(depart_time)
    exposure = 0.0; cnt = 0
    for (lat,lon), dtsec in zip(samples, timesec):
        when = (depart + pd.to_timedelta(int(dtsec//3600)*1,'H'))
        sub = df[df['time']==when]
        if sub.empty: continue
        idx = ((sub['lat']-lat)**2 + (sub['lon']-lon)**2).idxmin()
        exposure += float(sub.loc[idx,'risk']); cnt += 1
    return (exposure / max(1,cnt))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='data_sample/predictions.csv')
    ap.add_argument('--route', default='data_sample/route.json')
    ap.add_argument('--depart', required=True)
    ap.add_argument('--sweep', type=int, default=6)
    args = ap.parse_args()
    base_depart = pd.to_datetime(args.depart)
    scores = []
    for h in range(-args.sweep, args.sweep+1):
        depart = (base_depart + pd.to_timedelta(h,'H')).isoformat()
        s = exposure_for_departure(args.preds, args.route, depart)
        scores.append((h,s))
    best = min(scores, key=lambda x: x[1])
    print('Shift (h), exposure')
    for h,s in scores: print(h, round(s,4))
    print('Best shift:', best[0], 'hours (exposure=', round(best[1],4), ')')
