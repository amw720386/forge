"""Correct metadata from saved runtime token logs; preserve original measurements."""
import json
import re
from pathlib import Path

root=Path('benchmarks/results/phase15')
log=(root/'accepted-context-profile-server.log').read_text(encoding='utf-8')
actual=[int(x) for x in re.findall(r'\[phase14 tokenize\] tokens=(\d+)',log)]
requested=[128,512,1024,2048,4096]
assert len(actual)==10 and all(actual[i*2]==actual[i*2+1] for i in range(5)),actual
mapping={body:actual[i*2] for i,body in enumerate(requested)}
assert all(mapping[n]==n+19 for n in requested),mapping
for label in ['context-baseline','context-final','context-profile-final',
              'accepted-context-baseline','accepted-context-final','accepted-context-profile']:
    path=root/(label+'.json')
    original=root/('uncorrected-'+path.name)
    if not original.exists():original.write_bytes(path.read_bytes())
    data=json.loads(original.read_text(encoding='utf-8'))
    for row in data['results']:
        body=row['actual_input_tokens']
        row['requested_body_tokens']=body
        row['actual_input_tokens']=mapping[body]
    data['input_count_audit']='Corrected from runtime tokenize logs: server.formatPrompt adds 19 tokens. Timings and wire prompts unchanged; original JSON retained under uncorrected- prefix.'
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
(root/'context-token-count-audit.json').write_text(json.dumps(dict(body_to_runtime_tokens=mapping,runtime_profile_token_counts=actual,source='accepted-context-profile-server.log',timings_changed=False),indent=2),encoding='utf-8')
print(mapping)
