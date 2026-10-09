"""Summarize repeated HTTP evidence without combining timing domains."""
import argparse,json,statistics
from pathlib import Path

def summarize(path):
    data=json.loads(path.read_text())
    rows=[]
    for cell in data['results']:
        trials=cell['trials']
        rates=[t['aggregate_tok_s'] for t in trials]
        def median(key):return statistics.median(t[key] for t in trials)
        stages={}
        names=set().union(*(t.get('server_stage_deltas',{}).keys() for t in trials))
        for name in names:
            values=[t['server_stage_deltas'][name]['mean_ms'] for t in trials if t['server_stage_deltas'].get(name,{}).get('mean_ms') is not None]
            if values:stages[name]=statistics.median(values)
        rows.append(dict(workload=cell['workload'],concurrency=cell['concurrency'],
            output_tok_s=statistics.median(rates),output_tok_s_min=min(rates),output_tok_s_max=max(rates),
            ttft_ms=median('ttft_p50_s')*1000,p50_ms=median('latency_p50_s')*1000,p95_ms=median('latency_p95_s')*1000,
            maximum_request_ttft_ms=max(r['ttft_s']*1000 for t in trials for r in t['requests'] if r['ttft_s'] is not None),
            request_throughput_s=statistics.median(len(t['requests'])/t['wall_s'] for t in trials),
            effective_decode_tok_s=statistics.median(sum(max(0,r['tokens']-1) for r in t['requests']) / max(r['latency_s']-r['ttft_s'] for r in t['requests']) for t in trials),
            kv_utilization_peak=max((t.get('observed_kv_utilization_peak') or 0) for t in trials),
            scheduled_tokens_peak=max((t.get('observed_scheduled_tokens_peak') or 0) for t in trials),
            actual_output_tokens=[sum(r['tokens'] for r in t['requests']) for t in trials],
            active_mean=median('observed_active_mean'),peak_vram_mib=max(t['peak_vram_mib_sampled'] or 0 for t in trials),stage_mean_ms=stages))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    result={}
    for path in sorted(args.root.glob('*-http.json')):
        try:result[path.stem]=summarize(path)
        except (ValueError,KeyError,statistics.StatisticsError):continue
    (args.root/'summary.json').write_text(json.dumps(result,indent=2))
    for name, rows in result.items():
        for r in rows:
            print(name,r['workload'],r['concurrency'],round(r['output_tok_s'],2),round(r['ttft_ms'],1),round(r['p95_ms'],1))
