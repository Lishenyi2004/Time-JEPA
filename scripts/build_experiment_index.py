"""Build documentation indexes from the published YAMLs; never launch experiments.

Uses only the current release configs and their encoder input paths.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,re,shlex
import yaml

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'release-manifest.json').read_text())
bundled={r['path'] for r in manifest}
metadata_policy=json.loads((ROOT/'metadata-policy.json').read_text())
local_metadata={r['path']:r for r in metadata_policy['restricted_metadata']}
config_paths=sorted(p for p in bundled if p.endswith('.yaml') and not p.startswith('configs/quickstart/'))
raw={p:yaml.safe_load((ROOT/p).read_text()) for p in config_paths}
resolved={};bases={}
def merge(a,b):
 out=dict(a)
 for k,v in b.items():out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
 return out

def resolve(path,stack=()):
 if path in resolved:return resolved[path]
 assert path not in stack,(path,stack)
 d=raw[path];key=next((k for k in ['base','base_config'] if d.get(k)),None);base={};deps=[]
 if key:
  value=d[key];candidates=[(ROOT/path).parent/value,ROOT/value]
  p=next(q.resolve() for q in candidates if q.is_file());bp=str(p.relative_to(ROOT));base=resolve(bp,stack+(path,));deps=[bp]+bases[bp]
 if key=='base_config':
  # The probe loaders flatten top-level 'train' and inherit argparse defaults.
  child=d.get('train',{}) if 'train' in d else {k:v for k,v in d.items() if k not in ['base_config','runtime','rjob']}
  if 'train' in base:base=base['train']
  result={**base,**child}
 else:result=merge(base,{k:v for k,v in d.items() if k!='base'})
 resolved[path]=result;bases[path]=list(dict.fromkeys(deps));return result
for p in config_paths:resolve(p)

def leaves(value,prefix=''):
 if isinstance(value,dict):
  for k,v in value.items():yield from leaves(v,prefix+'.'+k if prefix else k)
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from leaves(v,prefix+f'[{i}]')
 elif isinstance(value,str):yield prefix,value

def domains(path,d):
 if path.startswith('physics_consistency/'):
  name=path.split('/')[1];return ['agibot' if name=='agibot_reach_rate' else 'egoexo' if name=='egoexo_knee' else name]
 if path.startswith(('grasp_eval_rjob/','physion_ocp_rjob/')):return ['grasp' if path.startswith('grasp') else 'physion']
 found=[]
 if d.get('dataset') in ['covla','agibot','egoexo']:found.append(d['dataset'])
 if d.get('data',{}).get('domain') in ['covla','agibot','egoexo']:found.append(d['data']['domain'])
 for key,val in leaves(d):
  if val=='data/CoVLA-Dataset':found.append('covla')
  if val=='data/agibot':found.append('agibot')
  if val=='data/EgoExo4D_minimal':found.append('egoexo')
 if 'psg_jepa/' in path:found.append('agibot')
 return sorted(set(found))

def training(path):return path.startswith('configs/train_time_jepa/') or '/adaptation/' in path or path.startswith('external_baselines/psg_jepa/configs/agibot_')

def scope(path,d):
 if 'grasp_eval_rjob/' in path:return ('external_grasp','Section 7.1; Appendix H','GRASP violation of expectation')
 if 'physion_ocp_rjob/' in path:return ('external_physion','Section 7.2; Appendix H','Physion future contact')
 if 'psg_jepa/' in path:return ('alternative_psg','Appendix K.1','Individual-frame / frame-pair supervision')
 if 'covla_law_rjob/' in path or 'law_alpha' in path or path.endswith('/full/law.yaml'):return ('alternative_law','Appendix K.2','Explicit physical-law supervision')
 if 'world_model_probe_rjob/' in path:return ('additional_backbones','Section 5.1; Appendix B','Additional video backbones')
 if 'physics_consistency/' in path:return ('kinematic_reconstruction','Section 6; Appendices E/F','Kinematic reconstruction')
 if 'egoexo_head_sweep/' in path:return ('head_aggregation','Appendix I.1','Auxiliary-head temporal aggregation')
 if '/ablation_a3/' in path:return ('probe_capacity','Appendix I.2','Frozen-probe capacity')
 if ('m2d_' in path or '/full_misaligned/' in path or 'sensor_robustness/' in path):return ('sensor_sensitivity','Appendix J','Temporal alignment / sensor noise')
 if 'egoexo_ablation/' in path or '/ablation_a4/' in path:return ('supervision_breadth','Appendix I.3','Aligned supervision breadth')
 if 'future_state_1p6s_8step/' in path:return ('future_eight_horizons','Appendix G','Eight-horizon future state / persistence')
 if 'future_state_single_domain/' in path:return ('future_single_domain','Appendices F/G','Single-domain five-horizon future state')
 if 'future_state_1s/' in path:return ('future_five_horizons','Section 5.3; Appendix G','Five-horizon future state')
 if '/covla_egoexo/' in path:return ('matched_domains','Appendix F','Matched human-motion domain-addition sequence')
 if '/covla_agibot_egoexo/' in path:return ('multi_domain','Section 5.2; Appendix F','Three-domain training and current-state probes')
 return ('single_domain','Section 5.1; Appendix B','Single-domain current-state decoding')

models={};training_model={};resources={};checks=[]
def resource(kind,path,status=None,description=None,generator=None):
 rid='r_'+hashlib.sha256((kind+':'+path).encode()).hexdigest()[:12]
 generated=local_metadata.get(path)
 if generated is None and kind=='split_metadata' and path!='time-jepa/splits':
  generated=next((r for p,r in local_metadata.items() if p.startswith(path.rstrip('/')+'/')),None)
 if generated:
  status='generate';generator=generated['preparation_entrypoint']
 if status is None:
  status='included' if path in bundled or any(p.startswith(path.rstrip('/')+'/') for p in bundled) else 'generate' if kind=='gate_or_audit' else 'external'
 rec={'id':rid,'kind':kind,'path':path,'availability':status}
 if description:rec['description']=description
 if generator:rec['preparation_entrypoint']=generator
 if rid not in resources:resources[rid]=rec
 elif generator:resources[rid]['preparation_entrypoint']=generator
 return rid

DATA={'covla':('data/CoVLA-Dataset','time-jepa/splits/covla_weather_ood','time-jepa/make_covla_weather_ood_splits.py'), 'agibot':('data/agibot','time-jepa/splits','time-jepa/make_agibot_splits.py'), 'egoexo':('data/EgoExo4D_minimal','time-jepa/splits/egoexo_exo124_view3ood_split_manifest.json','time-jepa/make_egoexo_splits.py'),'grasp':('external_baselines/grasp/data/videos/level2',None,'grasp_eval_rjob/audit_data.py'),'physion':('data/physion_v1_5',None,'physion_ocp_rjob/check_environment.py')}
for p in config_paths:
 if not training(p):continue
 d=resolved[p];ds=domains(p,d);name=Path(p).stem
 if p.startswith('world_model_probe_rjob/'):
  family='ivideogpt';variant=d['optimization']['mode'];mid=f'ivideogpt_{ds[0]}_{variant}';label=f'iVideoGPT {ds[0]} '+('Time-JEPA' if variant=='time' else 'Video FT')
 elif 'psg_jepa/' in p:
  family='psg_jepa_vitg';variant='static' if 'static' in name else 'full';mid=f'psg_agibot_{variant}';label=f'PSG-JEPA ViT-g {variant}'
 else:
  family='vjepa2_1';variant='video_only' if 'video_only' in name else 'time';mid='vjepa_'+'_'.join(ds)+'_'+variant;label='V-JEPA '+', '.join(ds)+' '+('Video FT' if variant=='video_only' else 'Time-JEPA')
  if 'egoexo_head_sweep/' in p:mid='vjepa_egoexo_head_'+name;label='V-JEPA human motion head '+name
  elif 'egoexo_ablation/' in p:mid='vjepa_egoexo_'+name.removeprefix('egoexo_');label='V-JEPA human motion '+name.removeprefix('egoexo_')
  elif 'sensor_robustness/' in p:mid='vjepa_egoexo_'+name.removeprefix('egoexo_');label='V-JEPA human motion '+name.removeprefix('egoexo_')
  elif 'law_alpha' in p:mid='vjepa_covla_time_law';label='V-JEPA autonomous driving Time-JEPA + Law'
  elif '/covla_egoexo/egoexo_only' in p:mid='vjepa_egoexo_matched_time';label='V-JEPA human motion matched single-domain Time-JEPA'
 out=d.get('folder',d.get('runtime',{}).get('output_dir',d.get('output_dir')))
 assert mid not in models,mid
 models[mid]={'id':mid,'name':label,'family':family,'domains':ds,'variant':variant,'training_config':p,'training_output_dir':out,'expected_weight_paths':[],'paper_scope':scope(p,d)[1]}
 training_model[p]=mid
for mid,name,family,path in [
 ('vjepa_original','Original V-JEPA 2.1 ViT-g/16','vjepa2_1','vjepa2_1_vitg_384.pt'),
 ('internvideo_original','InternVideo-Next-L frozen','internvideo_next','InternVideo/checkpoints/internvideo_next_large_p14_res224_f16/model.safetensors'),
 ('videomaev2_original','VideoMAE V2-G frozen','videomaev2','VideoMAEv2/checkpoints/mae-g/vit_g_hybrid_pt_1200e.pth'),
 ('ivideogpt_original','iVideoGPT OXE frozen','ivideogpt','iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium/transformer/model.safetensors')]:
 models[mid]={'id':mid,'name':name,'family':family,'domains':[],'variant':'original','training_config':None,'expected_weight_paths':[path],'paper_scope':'Baseline / initialization'}

def identify(weight):
 for mid,m in models.items():
  if weight in m['expected_weight_paths']:return mid
 candidates=[]
 for mid,m in models.items():
  out=m.get('training_output_dir')
  if not out:continue
  if weight.startswith(out.rstrip('/')+'/'):candidates.append(mid)
 assert len(candidates)==1,('Cannot associate released encoder input with a training config',weight,candidates)
 mid=candidates[0];models[mid]['expected_weight_paths'].append(weight);return mid

experiments=[]
for path in config_paths:
 d=resolved[path];ds=domains(path,d);group,paper,task=scope(path,d)
 role='training' if training(path) else 'base_config' if Path(path).name.startswith('_base') else 'preparation' if path.endswith('/gt_gate.yaml') else 'evaluation'
 model_ids=[training_model[path]] if role=='training' else []
 weight_fields=[]
 for field,value in leaves(d):
  tail=field.split('.')[-1]
  if tail in ['encoder_checkpoint','checkpoint','adapted_transformer_checkpoint'] and role=='evaluation':
   model_ids.append(identify(value));weight_fields.append({'field':field,'path':value})
 model_ids=list(dict.fromkeys(model_ids))
 supporting_model_ids=[]
 if role=='evaluation' and d.get('adapted_transformer_checkpoint'):
  primary=identify(d['adapted_transformer_checkpoint'])
  supporting_model_ids=[mid for mid in model_ids if mid!=primary]
  model_ids=[primary]
 if role=='evaluation':assert model_ids,('Evaluation lacks an encoder reference',path)
 rids=[resource('environment','requirements.txt','included','Install the recorded jepa package versions')]
 for domain in ds:
  data,split,gen=DATA[domain];rids.append(resource('dataset',data,'external',f'{domain} raw dataset',gen))
  if split:rids.append(resource('split_metadata',split,'included',f'{domain} formal split metadata',gen))
  if domain=='agibot':rids.append(resource('pca','time-jepa/cache/agibot_wholebody_cartesian_v1_waist_pca.json','included','Frozen train-only waist PCA','time-jepa/fit_agibot_waist_pca.py'))
 for field,value in leaves(d):
  key=field.split('.')[-1];kind=None
  if key in ['encoder_checkpoint','checkpoint','adapted_transformer_checkpoint','read_checkpoint','checkpoint_path','base_snapshot']:kind='encoder_weight' if key!='base_snapshot' else 'pretrained_snapshot'
  elif key=='urdf':kind='urdf'
  elif 'gate' in key or key=='data_audit':
   if '/' in value:kind='gate_or_audit'
  elif key in ['stats_cache','state_stats_path','action_stats_path','cache'] or field.startswith('stats.') and '/' in value:kind='normalization'
  elif key in ['encoder_train_split','manifest_path'] or key.endswith('manifest_path'):kind='split_metadata'
  elif key in ['upstream_root','repo_root']:kind='source_checkout'
  elif key=='params_path':kind='input_config'
  if kind:rids.append(resource(kind,value,'external' if kind in ['encoder_weight','pretrained_snapshot','urdf','source_checkout'] else None))
 if path.startswith('world_model_probe_rjob/') and 'ivideogpt_oxe_64_medium' in path:
  rids.extend([resource('environment','requirements-ivideogpt.txt','included','Additional iVideoGPT dependencies'),resource('source_checkout','iVideoGPT','external','Pinned source; use scripts/setup_third_party.py','scripts/setup_third_party.py')])
 if 'psg_jepa/' in path:
  rids.extend([resource('environment','requirements-psg.txt','included','Additional PSG training dependencies'),resource('source_checkout','third_party/PSG-JEPA','external','Pinned PSG source','scripts/setup_third_party.py')])
 if 'videomaev2_g/' in path:rids.append(resource('source_checkout','VideoMAEv2','external','Pinned VideoMAE V2 source','scripts/setup_third_party.py'))
 if 'internvideo_next_l/' in path:rids.append(resource('snapshot_companions','InternVideo/checkpoints/internvideo_next_large_p14_res224_f16','external','Full pretrained snapshot with custom modeling and configuration Python files'))
 if 'ivideogpt_oxe_64_medium' in path:
  rids.append(resource('snapshot_companions','iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium','external','Full official tokenizer/transformer snapshot and configuration files'))
  if d.get('adapted_transformer_checkpoint'):
   parent=Path(d['adapted_transformer_checkpoint']).parent
   rids.append(resource('encoder_metadata',str(parent/'config.json'),'external','Adapted transformer configuration'))
   rids.append(resource('encoder_metadata',str(parent.parent/'adaptation_manifest.json'),'external','Adapter-required model metadata'))
 if role=='training':
  if 'world_model_probe_rjob/' in path:entries=['world_model_probe_rjob/train_ivideogpt_adapt.py'];command=f'python {entries[0]} --config {shlex.quote(path)}'
  elif 'psg_jepa/' in path:entries=['external_baselines/psg_jepa/train_vitg.py'];command=f'python -m external_baselines.psg_jepa.train_vitg --config {shlex.quote(path)}'
  else:
   entry='covla_law_rjob/train_from_yaml.py' if d.get('app')=='vjepa_2_1_law' else 'covla_jepa_rjob/train_from_yaml.py';entries=[entry];command=f'torchrun --standalone --nproc_per_node=8 {entry} --fname {shlex.quote(path)}'
 elif role=='base_config':entries=[];command=None
 elif path.startswith('physics_consistency/'):
  domain=path.split('/')[1]
  module=f'physics_consistency.{domain}.'+('gt_gate' if domain=='agibot_reach_rate' else 'gt_kinematic_gate') if role=='preparation' else f'physics_consistency.{domain}.'+('train_head_bank' if domain=='agibot_reach_rate' else 'train_independent_head_bank')
  entries=[module.replace('.','/')+'.py'];command=f'python -m {module} --config {shlex.quote(path)}'
 elif path.startswith('grasp_eval_rjob/'):
  entries=['grasp_eval_rjob/run_eval.py','grasp_eval_rjob/score.py'];command=f'python -m grasp_eval_rjob.run_eval --config {shlex.quote(path)}';rids.append(resource('gate_or_audit',d['data_audit'],'generate','Generate and freeze the GRASP data audit','grasp_eval_rjob/audit_data.py'))
 elif path.startswith('physion_ocp_rjob/'):
  entries=['physion_ocp_rjob/extract_features.py','physion_ocp_rjob/run_probe.py'];command=f'python -m physion_ocp_rjob.extract_features --config {shlex.quote(path)}'
 elif 'psg_jepa/' in path:entries=['external_baselines/psg_jepa/probe.py'];command=f'python -m external_baselines.psg_jepa.probe --config {shlex.quote(path)}'
 elif path.startswith('world_model_probe_rjob/'):
  entries=['world_model_probe_rjob/'+('train_ivideogpt_adapted_probe.py' if d.get('adapted_transformer_checkpoint') else 'train_ivideogpt_probe.py' if 'ivideogpt_oxe_64_medium' in path else 'train_world_model_probe.py')];command=f'python {entries[0]} --config {shlex.quote(path)}'
 else:entries=['covla_probe_rjob/train_covla_probe.py'];command=f'python {entries[0]} --config {shlex.quote(path)}'
 for entry in entries:assert entry in bundled,('Missing entrypoint',entry)
 out=d.get('folder',d.get('output_dir',d.get('runtime',{}).get('output_dir',d.get('output',{}).get('root',d.get('output',{}).get('dir')))))
 probe_task='future_state' if 'future_' in path or d.get('target_mode')=='future_state' else 'held_out_variables' if '/shot/' in path or 'shot' in Path(path).stem or 'unseen' in Path(path).stem else 'current_state' if 'readout' in path or 'seen' in Path(path).stem else task
 experiments.append({'id':'exp_'+hashlib.sha256(path.encode()).hexdigest()[:12],'group':group,'paper_location':paper,'role':role,'task':probe_task,'domains':ds,'config':path,'inherited_configs':bases[path],'encoder_ids':model_ids,'supporting_encoder_ids':supporting_model_ids,'encoder_inputs':weight_fields,'entrypoints':entries,'command_template':command,'output_dir':out,'resource_ids':sorted(set(rids))})
# Multi-stage commands and repeats are recorded as templates only.
for e in experiments:
 e['command_sequence']=[e['command_template']] if e['command_template'] else []
 if e['role']!='evaluation':continue
 path=e['config']
 if path.startswith('grasp_eval_rjob/'):
  e['command_sequence'].append(f"python -m grasp_eval_rjob.score --config {shlex.quote(path)}")
 elif path.startswith('physion_ocp_rjob/'):
  e['command_sequence'].extend(f"python -m physion_ocp_rjob.run_probe --config {shlex.quote(path)} --protocol {protocol}" for protocol in ['all_scenarios','all_but_one'])
 elif path.startswith('physics_consistency/'):
  domain=path.split('/')[1]
  aggregate='aggregate' if domain=='agibot_reach_rate' else 'aggregate_grounded'
  e['probe_seeds']=[0,1,2,3,4]
  e['aggregation_entrypoint']=f'physics_consistency/{domain}/{aggregate}.py'
  e['aggregation_command']=f'python -m physics_consistency.{domain}.{aggregate}'

# Make resource provenance more useful without reading historical checkpoint metadata.
for rid,rec in resources.items():
 if rec['kind']=='gate_or_audit':
  if 'covla_law' in rec['path']:rec['preparation_entrypoint']='covla_law_rjob/gate.py'
  elif 'physics_consistency/covla/' in rec['path']:rec['preparation_entrypoint']='physics_consistency/covla/covla_gt_consistency_gate.py'
  elif 'physics_consistency/agibot_reach_rate/' in rec['path']:rec['preparation_entrypoint']='physics_consistency/agibot_reach_rate/gt_gate.py'
  elif 'physics_consistency/egoexo_knee/' in rec['path']:rec['preparation_entrypoint']='physics_consistency/egoexo_knee/gt_kinematic_gate.py'
  elif 'physics_consistency/egoexo/' in rec['path']:rec['preparation_entrypoint']='physics_consistency/egoexo/gt_kinematic_gate.py'
for mid,m in models.items():
 m['expected_weight_paths']=sorted(set(m['expected_weight_paths']))
 assert m['expected_weight_paths'],('No evaluation references training model',mid)
 m['evaluation_configs']=[e['config'] for e in experiments if e['role']=='evaluation' and mid in e['encoder_ids']]
 if m.get('training_output_dir') and any(not w.startswith(m['training_output_dir'].rstrip('/')+'/') for w in m['expected_weight_paths']):
  m['path_note']='The configured training output and required evaluation input directory differ. Place the released encoder at the exact input path shown in each evaluation YAML; alternatively update those paths together.'
  checks.append({'model_id':mid,'kind':'output_input_directory_difference','training_output_dir':m['training_output_dir'],'expected_weight_paths':m['expected_weight_paths']})

doc=ROOT/'docs';data={'schema_version':1,'scope':'Current published YAML configurations only; no original-checkpoint or epoch-alias mapping.','encoders':list(models.values()),'experiments':experiments,'resources':list(resources.values()),'path_notes':checks}
(doc/'experiment-index.json').write_text(json.dumps(data,indent=2)+'\n')
fields=['paper_location','group','role','task','domains','config','encoder_ids','training_configs','entrypoints','required_encoder_paths','required_resources','output_dir','command_template','command_sequence']
with (doc/'experiment-index.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for e in experiments:
  writer.writerow({'paper_location':e['paper_location'],'group':e['group'],'role':e['role'],'task':e['task'],'domains':'; '.join(e['domains']),'config':e['config'],'encoder_ids':'; '.join(e['encoder_ids']),'training_configs':'; '.join(dict.fromkeys(models[mid]['training_config'] or 'Official pretrained model; no local training config' for mid in e['encoder_ids'])),'entrypoints':'; '.join(e['entrypoints']),'required_encoder_paths':'; '.join(x['path'] for x in e['encoder_inputs']),'required_resources':'; '.join(resources[r]['path']+' ['+resources[r]['availability']+']' for r in e['resource_ids']),'output_dir':e['output_dir'],'command_template':e['command_template'],'command_sequence':json.dumps(e['command_sequence'])})

def link(path):return f'[{path}](../{path})'
lines=['# Encoder and resource index','','This index records the encoder currently required by each published experiment. Weight paths are copied from the release YAMLs. All model weights and pretrained snapshots are external inputs.','','The exact per-configuration mappings are in [experiment-index.csv](experiment-index.csv) and [experiment-index.json](experiment-index.json).','','## Encoders','','| Encoder ID | Model | Training configuration | Expected evaluation weight path | Evaluation YAMLs |','|---|---|---|---|---|']
for mid,m in models.items():lines.append('| '+ ' | '.join([f'`{mid}`',m['name'],link(m['training_config']) if m['training_config'] else 'Official pretrained model','<br>'.join('`'+p+'`' for p in m['expected_weight_paths']),str(len(m['evaluation_configs']))])+' |')
lines+=['','## Resources','','`included` means the path belongs to the public release manifest, regardless of local links used for acceptance. `external` means data, weights, URDF or upstream sources must be obtained separately. `generate` means run the listed preparation entrypoint before evaluation.','','| Resource | Kind | Availability | Preparation / note |','|---|---|---|---|']
for r in sorted(resources.values(),key=lambda r:(r['kind'],r['path'])):
 lines.append('| '+ ' | '.join(['`'+r['path']+'`',r['kind'],r['availability'],(link(r['preparation_entrypoint']) if r.get('preparation_entrypoint') else r.get('description',''))])+' |')
lines+=['','## Path consistency notes','']
if not checks:
 lines.append('All indexed training output directories match their corresponding evaluation encoder input directories.')
for item in checks:
 lines+=['- `'+item['model_id']+'`: its training YAML writes to `'+item['training_output_dir']+'`, while evaluation expects '+', '.join('`'+p+'`' for p in item['expected_weight_paths'])+'. Supply the encoder at the configured evaluation path, or change those directory paths consistently.']
lines+=['','The current YAMLs are unchanged by indexing. Adapted iVideoGPT rows identify the adapted encoder as primary; the official snapshot is a supporting input, not a second compared model. For iVideoGPT, retain the snapshot configuration/tokenizer files and adaptation manifest required by its adapter. GRASP also needs the predictor and target encoder contained in its configured checkpoint. Dataset preparation and omitted URDF/GT-gate resources are described in [data.md](data.md); third-party installation is in [dependencies.md](dependencies.md).','']
(doc/'models.md').write_text('\n'.join(lines))
lines=['# Formal experiment index','','Every formal experiment YAML in the release manifest is represented below. Training, evaluation, resource preparation and inherited base configurations are distinguished by role. Public quickstart and local runtime acceptance YAMLs are excluded.','','- [CSV: exact configuration, encoder, training configuration and required resources](experiment-index.csv)','- [JSON: resolved inheritance, encoder registry and resource availability](experiment-index.json)','- [Encoder and resource catalog](models.md)','','Commands are templates for the corresponding entrypoint, not a launch request. This indexing step performs no training or evaluation. Check each entrypoint\'s additional options for scenario protocols, repeated probe seeds and aggregation.','','## Experiment groups','']
for group in dict.fromkeys(e['group'] for e in experiments):
 group_entries=[e for e in experiments if e['group']==group]
 lines+=['### '+group.replace('_',' ').title(),'','Paper location: '+', '.join(dict.fromkeys(e['paper_location'] for e in group_entries))+'.','','| Role / task | Domain | Configuration | Encoder ID | Training configuration |','|---|---|---|---|---|']
 for e in group_entries:
  mids=e['encoder_ids'];tconfigs=list(dict.fromkeys(models[m]['training_config'] for m in mids))
  lines.append('| '+' | '.join([e['role']+' / '+e['task'],', '.join(e['domains']),link(e['config']),'<br>'.join('`'+m+'`' for m in mids) or 'Resource/base configuration','<br>'.join(link(p) if p else 'Official pretrained model' for p in tconfigs) or '—'])+' |')
 lines.append('')
lines+=['## Commands and preparation','','For Physion, `command_sequence` includes feature extraction followed by both all-scenario and leave-one-scenario-out readouts. GRASP sequences include evaluation and property scoring. Kinematic rows record probe seeds 0–4 and their aggregation entrypoint; repeat the bank command with `--seed` for each recorded value before aggregation. Commands are not executed by this index.\n\nUse `resource_ids` in JSON, or `required_resources` in CSV, to see the exact dataset, split, normalization/PCA, weight and optional source paths for a configuration. Only release-manifest resources are marked included; local acceptance links do not change that status. Training output directories and evaluation encoder input directories are checked during indexing; see the path consistency notes in [models.md](models.md).','']
lines+=['## Regenerate the index','','```bash','python scripts/build_experiment_index.py','```','','The generator reads only release-manifest YAMLs and writes the four index documents. It does not import model weights or start jobs. After intentionally changing indexed files, refresh their release-manifest checksums before publication.','']
(doc/'experiments.md').write_text('\n'.join(lines))
print(json.dumps({'configs':len(experiments),'encoders':len(models),'resources':len(resources),'roles':dict(Counter(e['role'] for e in experiments)),'path_notes':checks},indent=2))
