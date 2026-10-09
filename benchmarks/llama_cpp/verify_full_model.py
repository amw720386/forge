"""Audit every GGUF tensor against the exact Forge safetensors source after dtype conversion."""
import hashlib,json,pathlib,re,sys
import numpy as np
from gguf import GGUFReader
from safetensors import safe_open
model,gguf,out=map(pathlib.Path,sys.argv[1:]); reader=GGUFReader(str(gguf)); tensors={t.name:t for t in reader.tensors}
def mapping(name):
 if name=='model.embed_tokens.weight':return 'token_embd.weight'
 if name=='model.norm.weight':return 'output_norm.weight'
 if name=='lm_head.weight':return 'output.weight'
 m=re.fullmatch(r'model.layers.(\d+).(.+)\.(weight|bias)',name)
 if not m:raise ValueError('unmapped tensor '+name)
 layer,part,kind=m.groups();parts={'input_layernorm':'attn_norm','post_attention_layernorm':'ffn_norm','self_attn.q_proj':'attn_q','self_attn.k_proj':'attn_k','self_attn.v_proj':'attn_v','self_attn.o_proj':'attn_output','mlp.gate_proj':'ffn_gate','mlp.up_proj':'ffn_up','mlp.down_proj':'ffn_down'}
 return f'blk.{layer}.{parts[part]}.{kind}'
checks=[];mapped=set()
with safe_open(str(model/'model.safetensors'),framework='pt',device='cpu') as source:
 for name in source.keys():
  target=mapping(name);mapped.add(target)
  if target not in tensors:checks.append(dict(hf=name,gguf=target,match=False,reason='missing'));continue
  actual=tensors[target].data
  if actual.dtype not in [np.dtype('float16'),np.dtype('float32')]:raise ValueError('unsupported GGUF precision')
  raw=source.get_tensor(name).float().numpy();expected=raw.astype(actual.dtype)
  match=actual.shape==expected.shape and np.array_equal(actual,expected)
  error=float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32)))) if actual.shape==expected.shape else None
  checks.append(dict(hf=name,gguf=target,match=bool(match),max_abs_error=error,shape=list(actual.shape),dtype=str(actual.dtype)))
extra=sorted(set(tensors)-mapped)
config=json.loads((model/'config.json').read_text());tok=json.loads((model/'tokenizer.json').read_text());tc=json.loads((model/'tokenizer_config.json').read_text())
vocab=dict(tok['model']['vocab']);vocab.update({t['content']:t['id'] for t in tok.get('added_tokens',[])})
gg_tokens=reader.fields['tokenizer.ggml.tokens'].contents();mismatches=[dict(id=i,hf=t,gguf=gg_tokens[i] if i<len(gg_tokens) else None) for t,i in vocab.items() if i>=len(gg_tokens) or gg_tokens[i]!=t]
def field(key):return reader.fields[key].contents() if key in reader.fields else None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
result=dict(scope='All source tensors, exact equality after GGUF storage dtype conversion; no permutations applied for official Qwen2 conversion',tensor_count=len(checks),all_tensors_match=all(c['match'] for c in checks) and not extra,extra_gguf_tensors=extra,tensors=checks,tokenizer_vocab_mismatches=mismatches,source_vocab_count=len(vocab),gguf_vocab_count=len(gg_tokens),chat_template_match=field('tokenizer.chat_template')==tc.get('chat_template'),hf_config=config,gguf_rope_base=field('qwen2.rope.freq_base'),gguf_eos=field('tokenizer.ggml.eos_token_id'),gguf_bos=field('tokenizer.ggml.bos_token_id'),hashes={p.name:sha(p) for p in [model/'model.safetensors',model/'config.json',model/'tokenizer.json',model/'tokenizer_config.json',gguf]})
meta=model/'.cache/huggingface/download/config.json.metadata';result['hf_revision']=meta.read_text().splitlines()[0] if meta.exists() else None
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['tensors','hf_config','hashes','tokenizer_vocab_mismatches']}));sys.exit(0 if result['all_tensors_match'] and not mismatches else 1)
