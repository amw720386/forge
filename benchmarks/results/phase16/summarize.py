"""Summarize the retained unchanged Phase 15 HTTP harness evidence."""
import json, statistics
from pathlib import Path
root=Path(__file__).resolve().parent
def cells(name):
    return {(r['workload'],r['concurrency']):r for r in json.loads((root/name).read_text())['results']}
before={**cells('phase16-before-short-c1-http.json'),**cells('phase16-before-c8-http.json')}
after={**cells('phase16-after-short-c1-http.json'),**cells('phase16-after-c8-http.json')}
repeat=cells('phase16-after-short-repeat-http.json')
rows=[]
for key, baseline in before.items():
    candidate=after[key]
    def stats(cell):
        trials=cell['trials']
        return {'throughput_median':cell['aggregate_tok_s_median'],
                'throughput_range':[min(t['aggregate_tok_s'] for t in trials),max(t['aggregate_tok_s'] for t in trials)],
                'ttft_p50_ms_median':statistics.median(t['ttft_p50_s']*1000 for t in trials),
                'p95_ms_median':statistics.median(t['latency_p95_s']*1000 for t in trials),
                'peak_vram_mib':max(t.get('peak_vram_mib_sampled') or 0 for t in trials),
                'trials':len(trials)}
    row={'workload':key[0],'concurrency':key[1],'before':stats(baseline),'after':stats(candidate),
         'change_percent':100*(candidate['aggregate_tok_s_median']/baseline['aggregate_tok_s_median']-1)}
    if key in repeat:row['repeat']=stats(repeat[key])
    rows.append(row)
audit=[]
for label, collection in [('after',after),('repeat',repeat)]:
    for key, cell in collection.items():
        controls=before[key]['trials'][0]['requests'];total=0;differences=[]
        for i,trial in enumerate(cell['trials']):
            for j,request in enumerate(trial['requests']):
                total+=1
                if request['text']!=controls[j]['text']:differences.append({'trial':i,'request':j,'before':controls[j]['text'],'after':request['text']})
        audit.append({'label':label,'workload':key[0],'concurrency':key[1],'requests':total,'exact_matches':total-len(differences),'differences':differences})
result={'rows':rows,'output_audit':audit,'matching_completions':sum(r['exact_matches'] for r in audit),'completions':sum(r['requests'] for r in audit)}
(root/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
