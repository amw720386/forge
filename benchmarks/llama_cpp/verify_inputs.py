import json,pathlib,urllib.request,sys,time
from tokenizers import Tokenizer
sys.path.insert(0,str(pathlib.Path('/teamspace/studios/this_studio/forge-current/benchmarks/forge')))
from bench_http import PROMPTS
hf=Tokenizer.from_file('/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct/tokenizer.json')
rows=[]
for name,prompt in PROMPTS.items():
 payload=dict(content=prompt,add_special=True,parse_special=True)
 req=urllib.request.Request('http://localhost:8080/tokenize',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:actual=json.load(r)['tokens']
 expected=hf.encode(prompt,add_special_tokens=True).ids
 rows.append(dict(workload=name,count=len(expected),match=expected==actual,hf_ids=expected,llama_ids=actual))
out=pathlib.Path('/teamspace/studios/this_studio/forge-cache/phase14-results/input-identity.json');out.write_text(json.dumps(rows,indent=2))
print(json.dumps([dict(workload=r['workload'],count=r['count'],match=r['match']) for r in rows]))
assert all(r['match'] for r in rows)
