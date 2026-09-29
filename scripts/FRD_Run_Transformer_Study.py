# -*- coding: utf-8 -*-
"""FRD Transformer study: environment, split, training, selection and ensembles.

One file for every Transformer step of the study. It contains, in this order:

  1. process supervisor  - runs each training task in a separate process,
                           verifies its outputs, and regenerates predictions
                           from saved weights when they are missing
  2. no-large study      - plain Transformers, BERT sentiment variants,
                           FRD -> OpSpam zero-shot test
  3. sentiment extension - remaining sentiment variants, backbone
                           representatives, 35-triple ensemble search
  4. environment setup   - pinned PyTorch/Transformers environment
  5. plain DeBERTa-v3    - trained in FP32 by a separate routine
  6. command line
  7. embedded runtime    - data split, model registry, training loop,
                           metrics and threshold rule (verified zip archive)

Commands (run from the folder that holds the data files)
  python FRD_Run_Transformer_Study.py setup        create .venv_frd_safe (system Python 3.9-3.12)
  .venv_frd_safe python FRD_Run_Transformer_Study.py split       duplicate-safe split, no training
  .venv_frd_safe python FRD_Run_Transformer_Study.py deberta-v3  plain DeBERTa-v3 (skipped if present)
  .venv_frd_safe python FRD_Run_Transformer_Study.py train       all Transformer tasks
  .venv_frd_safe python FRD_Run_Transformer_Study.py all         split, deberta-v3, train
  .venv_frd_safe python FRD_Run_Transformer_Study.py status      read-only check of saved outputs

Every step reuses verified outputs, so running it on a finished results folder
only checks it. Required files: "fake reviews dataset.csv" and
"deceptive-opinion.csv". Results are written to ./results.
"""
from __future__ import annotations

import argparse
import base64
import csv
import faulthandler
import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path, PurePosixPath

SELF = Path(__file__).resolve()

# =============================================================================
# 1. Process supervisor (formerly FRD_Run_All_Safe.py v1.1.0; full-queue mode removed)
# =============================================================================
# -*- coding: utf-8 -*-


VERSION = '1.1.0'
WEIGHT_NAMES = ('model.safetensors', 'pytorch_model.bin', 'adapter_model.safetensors',
                'adapter_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')
REQUIRED_METRICS = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'mcc'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]


def atomic_json(path, obj, attempts=12):
    """Atomically replace JSON, tolerating transient Windows scanner/file locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    last = None
    try:
        for i in range(max(1, int(attempts))):
            try:
                os.replace(str(tmp), str(path))
                return
            except PermissionError as exc:
                last = exc
                if i + 1 >= attempts:
                    raise
                time.sleep(min(0.8, 0.025 * (2 ** i)))
        if last is not None:
            raise last
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def should_enable_periodic_traceback(platform_name=None):
    """Periodic all-thread dumps are unsafe around long CUDA kernels on Windows."""
    name = sys.platform if platform_name is None else str(platform_name)
    return not name.lower().startswith('win')


def snapshot_dataset_columns(frame, text_col, sentiment=None):
    """Materialize hot-loop dataset columns outside pandas' C-extension row accessor."""
    texts = [str(x) for x in frame[text_col].tolist()]
    labels = [int(x) for x in frame['label_num'].tolist()]
    if sentiment is None:
        sent = None
    else:
        import numpy as np
        sent = np.asarray(sentiment, dtype=np.float32).copy()
        if len(sent) != len(texts):
            raise ValueError('Sentiment feature rows do not match dataset rows.')
    return texts, labels, sent


def initial_profile_index(task_key, vram_gb):
    """Skip full-finetuning profiles that are structurally impossible on a 6 GiB GPU."""
    large = {'deberta_v3_large','modernbert_large','electra_large','roberta_large'}
    m = re.fullmatch(r'(?:frd|opspam):seed\d+:single:([A-Za-z0-9_]+)', str(task_key))
    model_key = m.group(1) if m else None
    try:
        low_vram = vram_gb is not None and float(vram_gb) <= 6.5
    except (TypeError, ValueError):
        low_vram = False
    # On <=6.5 GiB the original large profile list is:
    # 0 full_primary, 1 full_checkpointed, 2 peft_lora.
    return 2 if low_vram and model_key in large else 0


def detect_vram_gb():
    """Read total VRAM without importing torch/CUDA in the coordinator process."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
        first = str(out).strip().splitlines()[0].strip()
        mib = float(first)
        return mib / 1024.0
    except Exception:
        return None


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return default


def exit_hex(code):
    return '0x%08X' % (int(code) & 0xFFFFFFFF)


def absolute_path(project, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (Path(project) / p).resolve()


def extract_verified_zip(raw, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        if sum(x.file_size for x in members) > 100 * 1024 * 1024:
            raise ValueError('Embedded SOURCE archive exceeds 100 MiB; refusing extraction.')
        for m in members:
            name = m.filename.replace('\\', '/')
            p = PurePosixPath(name)
            if (p.is_absolute() or '..' in p.parts or ':' in name or
                    (m.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe embedded archive member: ' + m.filename)
            dest = (target / name).resolve()
            if dest != target and target not in dest.parents:
                raise ValueError('Embedded archive escapes target directory.')
        target.mkdir(parents=True, exist_ok=True)
        for m in members:
            dest = target / m.filename.replace('\\', '/')
            if m.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            content = z.read(m)
            if dest.exists() and dest.read_bytes() != content:
                raise ValueError('Immutable snapshot was edited: ' + str(dest))
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)


def original_runtime(project):
    """Extract the embedded runtime (PAYLOAD_B64, end of file) after checking its SHA-256."""
    project = Path(project).resolve()
    raw = base64.b64decode(PAYLOAD_B64.encode('ascii'), validate=False)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PAYLOAD_SHA256:
        raise ValueError('Embedded payload SHA256 mismatch; no runtime was changed.')
    target = project / '.frd_supervisor' / 'runtime' / digest
    extract_verified_zip(raw, target)
    for name in ('config.py','run_all.py','frdexp/training.py'):
        if not (target / name).is_file():
            raise FileNotFoundError('Original payload missing ' + name)
    return target, digest


def backup_database(source, dest):
    """SQLite online backup includes committed data still in the WAL file."""
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if not source.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=20)
    dst = sqlite3.connect(str(dest), timeout=20)
    try:
        src.backup(dst)
    finally:
        dst.close(); src.close()
    return True


def training_root(project, task_key):
    match = re.fullmatch(r'(frd|opspam):seed(\d+):(single|hybrid):([A-Za-z0-9_]+)', task_key)
    if not match:
        return None
    return Path(project) / 'results' / 'models' / match[1] / ('seed_' + match[2]) / match[4]


def has_weights(artifact):
    artifact = Path(artifact)
    for name in WEIGHT_NAMES:
        p = artifact / name
        if p.is_file() and p.stat().st_size > 0:
            if name.endswith('.index.json'):
                index = read_json(p, {})
                shards = set(index.get('weight_map', {}).values())
                if not shards or not all((artifact/s).is_file() and (artifact/s).stat().st_size for s in shards):
                    continue
            return True
    # Hybrid layouts are defined by the original runtime, not re-created here.
    meta = read_json(artifact/'artifact_meta.json', {}) or {}
    if meta.get('type') in {'hybrid', 'sentiment_fusion'}:
        return any(p.stat().st_size > 0 for p in artifact.rglob('*')
                   if p.is_file() and p.suffix in {'.pt','.bin','.safetensors'}
                   and p.name not in {'training_args.bin','optimizer.pt','scheduler.pt'})
    return False


def split_id_field(columns):
    """Return the persisted row identifier used by the original FRD runtime.

    The embedded runtime writes ``source_index`` to split CSVs and later writes
    that value as ``sample_id`` in prediction CSVs.  Some repaired/newer splits
    may already contain ``sample_id``.  Accept either without re-splitting.
    """
    columns = set(() if columns is None else columns)
    if 'sample_id' in columns:
        return 'sample_id'
    if 'source_index' in columns:
        return 'source_index'
    return None


def expected_partition(project, protocol, part):
    """Use only an existing persisted split, never make a new one."""
    path = Path(project)/'results'/'splits'/(protocol + '_split.csv')
    if not path.is_file():
        return None
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        id_field = split_id_field(columns)
        if id_field is None or not {'label_num','split'} <= columns:
            return None
        return [(str(r[id_field]), int(r['label_num'])) for r in reader if r['split'] == part]


def check_prediction_csv(path, expected=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'sample_id','y_true','y_prob_cg'} <= set(reader.fieldnames or []):
            raise ValueError('Prediction columns missing: ' + str(path))
        pairs, seen = [], set()
        for row in reader:
            sid = str(row['sample_id'])
            y = int(row['y_true'])
            p = float(row['y_prob_cg'])
            if sid in seen or y not in (0,1) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid/duplicate prediction row: ' + str(path))
            seen.add(sid); pairs.append((sid,y))
        if not pairs:
            raise ValueError('Empty prediction file: ' + str(path))
        if expected is not None and pairs != expected:
            raise ValueError('Prediction IDs/labels/order differ from saved split: ' + str(path))
        return len(pairs)


def inspect_training_bundle(project, task_key):
    project = Path(project).resolve()
    root = training_root(project, task_key)
    result = {'complete':False,'saved_model':False,'repairable':False,'issues':[]}
    if root is None:
        result['issues'].append('not_a_training_task')
        return result
    result['root'] = str(root)
    selected = root/'selected_result.json'
    payload = read_json(selected)
    if payload is None:
        candidates = sorted(root.glob('*/selected_result.json')) if root.exists() else []
        if len(candidates) == 1:
            selected = candidates[0]; payload = read_json(selected)
    if not isinstance(payload, dict):
        # Preserve a saved artifact even when selection JSON was not written.
        candidates = [p for p in root.glob('*/artifact') if has_weights(p)] if root.exists() else []
        if len(candidates) == 1:
            artifact = candidates[0]
            meta = read_json(artifact/'artifact_meta.json', {}) or {}
            metrics = read_json(artifact.parent/'metrics.json', {}) or {}
            payload = dict(metrics)
            payload.setdefault('model', meta.get('model_key', root.name))
            payload.setdefault('seed', int(root.parent.name.removeprefix('seed_')))
            payload.setdefault('resource', meta.get('resource', {}))
            payload['artifact_path'] = str(artifact)
            payload['result_dir'] = str(artifact.parent)
        else:
            result['issues'].append('no_selection_or_ambiguous_saved_artifacts')
            result['saved_model'] = bool(candidates)
            return result
    artifact = absolute_path(project, payload.get('artifact_path', str(root/'full_primary/artifact')))
    attempt = absolute_path(project, payload.get('result_dir', str(artifact.parent)))
    if root.resolve() not in artifact.parents or root.resolve() not in attempt.parents:
        result['issues'].append('artifact/result path is outside this model task')
        return result
    result.update(artifact=str(artifact), attempt=str(attempt), selected=str(selected), payload=payload)
    result['saved_model'] = has_weights(artifact)
    meta = read_json(artifact/'artifact_meta.json', {}) or {}
    result['repairable'] = bool(result['saved_model'] and ':single:' in task_key
                               and (artifact/'config.json').is_file()
                               and (artifact/'tokenizer/tokenizer_config.json').is_file()
                               and isinstance(payload.get('resource'), dict)
                               and payload['resource'].get('max_length'))
    if not result['saved_model']:
        result['issues'].append('model weights missing')
    if not (artifact/'artifact_meta.json').is_file():
        result['issues'].append('artifact_meta.json missing')
    if not (artifact/'tokenizer/tokenizer_config.json').is_file():
        result['issues'].append('saved tokenizer missing')
    if payload.get('model') != root.name or int(payload.get('seed',-1)) != int(root.parent.name[5:]):
        result['issues'].append('model key or seed mismatch')
        result['repairable'] = False
    metrics = read_json(attempt/'metrics.json')
    if not isinstance(metrics,dict):
        result['issues'].append('metrics.json missing/invalid')
        metrics = {}
    for source_name, source in [('selected_result',payload),('metrics',metrics)]:
        for section in ('validation_default','test_default','test_calibrated'):
            values = source.get(section,{})
            if not isinstance(values,dict) or not REQUIRED_METRICS <= set(values):
                result['issues'].append(source_name + ':' + section + ' metrics incomplete')
            elif any(not isinstance(values[k],(int,float)) or not math.isfinite(values[k]) for k in REQUIRED_METRICS):
                result['issues'].append(source_name + ':' + section + ' metrics invalid')
    th = payload.get('selected_threshold')
    if not isinstance(th,(int,float)) or not math.isfinite(th) or not 0 <= th <= 1:
        result['issues'].append('invalid selected threshold')
    protocol = task_key.split(':')[0]
    for part in ('validation','test'):
        p = attempt/(part+'_predictions.csv')
        try:
            count = check_prediction_csv(p, expected_partition(project,protocol,part))
            result[part+'_rows'] = count
        except (OSError,ValueError,KeyError) as exc:
            result['issues'].append(str(exc))
    if not (root/'selected_result.json').is_file():
        result['issues'].append('selection JSON not at task root')
    result['complete'] = not result['issues']
    return result


def run_logged(command, cwd, log_path, echo=True, env=None, timeout=0):
    """Keep stdout, stderr AND native faulthandler output, independent of return code."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('wb') as log:
        p = subprocess.Popen([str(x) for x in command], cwd=str(cwd), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             bufsize=0)
        def pump():
            while True:
                chunk = p.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk); log.flush()
                if echo:
                    try:
                        sys.stdout.write(chunk.decode('utf-8',errors='replace')); sys.stdout.flush()
                    except (BrokenPipeError,UnicodeError):
                        pass
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            try:
                code = p.wait(timeout=float(timeout) if timeout else None)
            except subprocess.TimeoutExpired:
                p.terminate()
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
                code = 124
            except KeyboardInterrupt:
                p.terminate()
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
                raise
        finally:
            thread.join(timeout=15)
            p.stdout.close()
        return int(code)


def worker_env(project, base=None):
    env = dict(os.environ if base is None else base)
    env.update(FRD_CSV=str(Path(project)/'fake reviews dataset.csv'),
               OPSPAM_CSV=str(Path(project)/'deceptive-opinion.csv'),
               FRD_RESULTS_DIR=str(Path(project)/'results'),
               USE_TORCH='1', USE_TF='0', USE_FLAX='0',
               TOKENIZERS_PARALLELISM='false', PYTHONFAULTHANDLER='1',
               PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return env


def stage(job, label, **extra):
    path = Path(job['directory'])/'stage.json'
    old = read_json(path,{}) or {}
    old.update(phase=label, time=now(), **extra)
    atomic_json(path, old)
    print('[STAGE] ' + label, flush=True)


class ProjectLock:
    def __init__(self,path): self.path=Path(path); self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('a+b')
        self.file.seek(0); self.file.write(b'0'); self.file.flush(); self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('Another FRD_Run_Transformer_Study process holds this project lock.')
        return self
    def __exit__(self,*args):
        if self.file:
            self.file.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
            self.file.close()


def preload_for_training(job):
    """Mitigate import-order interactions without changing tokenizer/training settings.

    This does NOT remove sklearn from Transformers. It loads the target objects
    first and passes them to the original trainer rather than loading them again.
    """
    payload = job.get('payload') or {}
    hf_id = payload.get('hf_id') or payload.get('encoder')
    key = job['task_key']
    if training_root(Path(job['project']),key) is None or not hf_id:
        return None
    stage(job,'preload:torch')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this interpreter; packages were NOT changed.')
    from config import CONFIG
    from frdexp.reproducibility import set_global_determinism
    seed = int(payload.get('seed', key.split(':')[1][4:]))
    set_global_determinism(seed, CONFIG.strict_deterministic)
    stage(job,'preload:transformers')
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    cache = str(CONFIG.model_cache_dir) if CONFIG.model_cache_dir else None
    saved = inspect_training_bundle(Path(job['project']),key)
    repair = saved.get('saved_model') and not saved.get('complete')
    if repair:
        if not saved.get('repairable'):
            raise RuntimeError('Saved model exists but automatic reconciliation is unsafe: ' + '; '.join(saved['issues']))
        model_source = saved['artifact']
        token_source = str(Path(model_source)/'tokenizer')
        tc = read_json(Path(token_source)/'tokenizer_config.json',{}) or {}
        # Respect the saved tokenizer class, including the earlier slow tokenizer.
        token_class = str(tc.get('tokenizer_class',''))
        use_fast = token_class.endswith('Fast') if token_class else (Path(token_source)/'tokenizer.json').exists()
        tokenizer_kwargs = dict(local_files_only=True,use_fast=use_fast)
        model_kwargs = dict(local_files_only=True)
        model_class = AutoModelForSequenceClassification
    else:
        model_source=token_source=hf_id
        tokenizer_kwargs=dict(cache_dir=cache,use_fast=True)
        model_class = AutoModel if ':hybrid:' in key else AutoModelForSequenceClassification
        model_kwargs=dict(cache_dir=cache)
        if ':single:' in key: model_kwargs['num_labels']=2
    objects={'hf_id':hf_id,'repair':bool(repair),'saved':saved,
             'tokenizer':None,'model':None,'model_class':model_class.__name__}
    def load_tokenizer():
        stage(job,'tokenizer:loading',source=token_source,tokenizer_kwargs=tokenizer_kwargs)
        objects['tokenizer']=AutoTokenizer.from_pretrained(token_source,**tokenizer_kwargs)
        stage(job,'tokenizer:ready',is_fast=bool(getattr(objects['tokenizer'],'is_fast',False)))
    def load_model():
        stage(job,'model:loading_cpu',source=model_source)
        objects['model']=model_class.from_pretrained(model_source,**model_kwargs)
        stage(job,'model:ready_cpu')
    # Retry changes ONLY initialization order, not hyperparameters or tokenizer type.
    if job.get('load_order')=='model_first':
        load_model(); load_tokenizer()
    else:
        load_tokenizer(); load_model()
    return objects


def reconcile_single(job, objects, spec, frame, text_col, seed, config, task_root):
    """Regenerate missing outputs from SAVED weights, never call train()."""
    import numpy as np
    import torch
    import frdexp.training as training
    from frdexp.metrics import choose_threshold
    saved=inspect_training_bundle(Path(job['project']),job['task_key'])
    if not saved.get('repairable'):
        raise RuntimeError('Saved artifact needs manual review; refusing to re-train it automatically.')
    if not objects or not objects.get('repair') or objects.get('model') is None:
        raise RuntimeError('Reconciliation model was not preloaded; refusing a guessed reload.')
    model,tokenizer=objects.pop('model'),objects.pop('tokenizer')
    old=dict(saved['payload']); resource=dict(old.get('resource') or {})
    max_length=int(resource['max_length'])
    val=frame[frame['split']=='validation'].copy()
    test=frame[frame['split']=='test'].copy()
    if not len(val) or not len(test):
        raise RuntimeError('Saved model reconciliation requires validation AND test splits.')
    stage(job,'reconcile:inference_only',artifact=saved['artifact'],max_length=max_length,
          note='No optimizer, training or new split; reference inference in FP32, batch size 1.')
    model.to('cuda'); model.eval()
    def predict(part):
        probs=[]
        with torch.no_grad():
            for i,text in enumerate(part[text_col]):
                inputs=tokenizer(str(text),truncation=True,max_length=max_length,
                                 padding=True,return_tensors='pt')
                inputs={k:v.to('cuda') for k,v in inputs.items()}
                logits=model(**inputs).logits
                probs.append(float(torch.softmax(logits.float(),dim=-1)[0,1].cpu()))
                if (i+1)%1000==0: print('[RECONCILE] predictions:',i+1,'/',len(part),flush=True)
        return np.asarray(probs,dtype=np.float64)
    start=time.perf_counter()
    vp,tp=predict(val),predict(test)
    threshold=choose_threshold(val.label_num,vp)
    attempt=Path(saved['attempt']); artifact=Path(saved['artifact'])
    # Preserve the previous record of training; do not claim a missing seed can be recovered.
    notes=['Outputs reconciled using existing weights and original metrics/threshold functions.',
           'Earlier standalone training settings remain legacy; do not assume identical training protocol.',
           'No claim that initialization seed was set before the earlier model/head was created.']
    # Actual saved TrainingArguments can correct the earlier large launcher's literal fp32 metadata.
    actual_args=artifact/'training_args.bin'
    if actual_args.is_file():
        try:
            args=torch.load(str(actual_args),map_location='cpu',weights_only=False)
            resource['precision']=('bf16' if getattr(args,'bf16',False) else
                                   'fp16' if getattr(args,'fp16',False) else 'fp32')
        except Exception as exc:
            notes.append('Could not verify stored TrainingArguments: '+repr(exc))
    provenance={'time':now(),'source':'saved_artifact_inference_only','supervisor_version':VERSION,
                'inference_precision':'fp32','inference_batch_size':1,
                'inference_seconds':time.perf_counter()-start,'legacy_notes':notes,
                'prior_resource':old.get('resource',{})}
    staging=attempt/('.reconcile_'+uuid.uuid4().hex)
    stage(job,'reconcile:original_metrics')
    result=training._write_metrics_bundle(staging,spec.key,seed,val,test,vp,tp,resource,threshold)
    result.update(artifact_path=str(artifact.resolve()),result_dir=str(attempt.resolve()),
                  profile=old.get('profile',attempt.name),reconciliation=provenance)
    atomic_json(staging/'metrics.json',result)
    # Validate row identity before publishing output files.
    for part in ('validation','test'):
        subset = val if part=='validation' else test
        id_field = split_id_field(subset.columns)
        if id_field is None:
            raise RuntimeError('Persisted split has no sample_id/source_index during reconciliation.')
        expected=[(str(sid),int(y)) for sid,y in zip(subset[id_field],subset.label_num)]
        check_prediction_csv(staging/(part+'_predictions.csv'),expected)
    backup=attempt/'reconciliation_backups'/stamp(); backup.mkdir(parents=True)
    for p in (attempt/'metrics.json',attempt/'validation_predictions.csv',attempt/'test_predictions.csv',
              Path(task_root)/'selected_result.json'):
        if p.is_file(): shutil.copy2(str(p),str(backup/p.name))
    for name in ('metrics.json','validation_predictions.csv','test_predictions.csv'):
        os.replace(str(staging/name),str(attempt/name))
    atomic_json(Path(task_root)/'selected_result.json',result)
    atomic_json(attempt/'reconciliation.json',provenance)
    stage(job,'reconcile:complete')
    return result


def peft_modules_to_save(model, original_names):
    """Preserve DeBERTa's freshly initialized pooling head in an adapter artifact.

    The base checkpoint has no task-specific pooler weights. A frozen randomly
    initialized pooler would otherwise be regenerated on adapter reload.
    Other model families keep the original module-selection contract.
    """
    names = list(original_names or [])
    model_type = str(getattr(getattr(model, 'config', None), 'model_type', ''))
    if model_type in {'deberta', 'deberta-v2'} and getattr(model, 'pooler', None) is not None:
        if 'pooler' not in names:
            names.append('pooler')
    return names or None


def prepare_peft_checkpointing(model):
    """Make checkpoint inputs differentiable without unfreezing base embeddings.

    Original runtime wraps PEFT before enabling gradient checkpointing. PEFT
    0.14 only prepares input gradients automatically when checkpointing was
    already enabled during construction. Use its public input-gradient API.
    """
    method = getattr(model, 'enable_input_require_grads', None)
    if not callable(method):
        raise RuntimeError('PEFT checkpointing requires enable_input_require_grads; refusing silent adapter detachment.')
    method()
    named = list(model.named_parameters())
    adapters = [(name, param) for name, param in named
                if 'lora_' in name and param.requires_grad]
    if not adapters:
        raise RuntimeError('No trainable LoRA parameters found; refusing a head-only run labeled as LoRA.')
    return {'input_gradient_hook': True,
            'trainable_adapter_tensors': len(adapters),
            'trainable_parameters': sum(p.numel() for _, p in named if p.requires_grad),
            'total_parameters': sum(p.numel() for _, p in named)}


def install_training_hooks(job, objects):
    """In-memory hooks only. A fresh process executes ONE original OOM profile."""
    import frdexp.training as training
    if not hasattr(training,'_hf_imports'):
        return  # Used only by the test runtime; real payload has this interface.
    # Avoid pandas .iloc in the long-running DataLoader hot loop. Several prior
    # Windows failures occurred exactly while pandas Series internals were being
    # accessed after long CUDA runs. Snapshot plain Python/NumPy values once.
    if hasattr(training, 'TextDataset'):
        import numpy as np
        from torch.utils.data import Dataset
        class SafeTextDataset(Dataset):
            def __init__(self, frame, tokenizer, text_col, max_length, sentiment=None):
                self.texts, self.labels, self.sentiment = snapshot_dataset_columns(
                    frame, text_col, sentiment)
                self.tokenizer = tokenizer
                self.max_length = max_length
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, i):
                enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
                enc['labels'] = self.labels[i]
                if self.sentiment is not None:
                    enc['sentiment_features'] = np.asarray(self.sentiment[i], dtype=np.float32)
                return enc
        training.TextDataset = SafeTextDataset
        stage(job, 'dataset:snapshot_mode')
    # Preserve the original LoRA configuration; only fix checkpoint gradient flow
    # and persistence of the randomly initialized DeBERTa pooling head.
    if hasattr(training, '_apply_peft') and hasattr(training, 'classifier_modules_to_save'):
        original_modules_to_save = training.classifier_modules_to_save
        def modules_to_save(model):
            return peft_modules_to_save(model, original_modules_to_save(model))
        training.classifier_modules_to_save = modules_to_save
        original_apply_peft = training._apply_peft
        def apply_peft(model):
            selected_modules = modules_to_save(model)
            out = original_apply_peft(model)
            info = prepare_peft_checkpointing(out)
            stage(job, 'peft:ready', modules_to_save=selected_modules, **info)
            print('[PEFT READY] trainable parameters:', info['trainable_parameters'],
                  '| saved modules:', selected_modules, flush=True)
            return out
        training._apply_peft = apply_peft
    hf_imports=training._hf_imports
    def observed_imports():
        values=list(hf_imports())
        tok_cls,seq_cls,enc_cls=values[:3]
        def proxy(factory,slot,accept):
            class Loader:
                @staticmethod
                def from_pretrained(source,*args,**kwargs):
                    if objects and not objects.get('repair') and str(source)==objects['hf_id'] and accept:
                        obj=objects.get(slot)
                        if obj is not None:
                            objects[slot]=None
                            stage(job,slot+':using_preloaded_object')
                            return obj
                    stage(job,slot+':original_loader',source=str(source))
                    return factory.from_pretrained(source,*args,**kwargs)
            return Loader
        values[0]=proxy(tok_cls,'tokenizer',True)
        values[1]=proxy(seq_cls,'model',':single:' in job['task_key'])
        values[2]=proxy(enc_cls,'model',':hybrid:' in job['task_key'])
        trainer_cls=values[4]
        class ObservedTrainer(trainer_cls):
            def __init__(self,*args,**kwargs):
                stage(job,'trainer:initializing')
                super().__init__(*args,**kwargs)
                stage(job,'trainer:ready')
            def train(self,*args,**kwargs):
                stage(job,'training:started')
                out=super().train(*args,**kwargs)
                stage(job,'training:finished')
                return out
            def predict(self,*args,**kwargs):
                stage(job,'prediction:started')
                out=super().predict(*args,**kwargs)
                stage(job,'prediction:finished')
                return out
        values[4]=ObservedTrainer
        return tuple(values)
    training._hf_imports=observed_imports
    original_profiles=training.build_training_profiles
    def one_profile(*args,**kwargs):
        import dataclasses
        profiles=original_profiles(*args,**kwargs)
        index=int(job.get('profile_index',0))
        descriptions=[dataclasses.asdict(p) for p in profiles]
        stage(job,'profile:selection',profiles=descriptions,profile_index=index,profile_count=len(profiles))
        if index>=len(profiles):
            raise RuntimeError('No remaining original training profile.')
        return [profiles[index]]
    training.build_training_profiles=one_profile
    original_oom=training._is_oom
    def record_oom(exc):
        found=original_oom(exc)
        if found: stage(job,'profile:out_of_memory',had_oom=True,error=repr(exc))
        return found
    training._is_oom=record_oom
    original_args=training._training_args
    def tracked_args(*args,**kwargs):
        out=original_args(*args,**kwargs)
        keys=('num_train_epochs','per_device_train_batch_size','per_device_eval_batch_size',
              'gradient_accumulation_steps','learning_rate','weight_decay','warmup_ratio',
              'seed','data_seed','fp16','bf16','dataloader_num_workers')
        stage(job,'training_arguments:ready',resolved_training_arguments={k:getattr(out,k,None) for k in keys})
        return out
    training._training_args=tracked_args
    original_single=training.train_single_model
    @functools.wraps(original_single)
    def guarded_single(spec,frame,text_col,seed,config,task_root):
        saved=inspect_training_bundle(Path(job['project']),job['task_key'])
        if saved.get('saved_model'):
            if saved.get('complete'): return saved['payload']
            return reconcile_single(job,objects,spec,frame,text_col,seed,config,task_root)
        return original_single(spec,frame,text_col,seed,config,task_root)
    training.train_single_model=guarded_single
    if hasattr(training,'train_hybrid_model'):
        original_hybrid=training.train_hybrid_model
        @functools.wraps(original_hybrid)
        def guarded_hybrid(*args,**kwargs):
            saved=inspect_training_bundle(Path(job['project']),job['task_key'])
            if saved.get('saved_model') and not saved.get('complete'):
                raise RuntimeError('Existing hybrid artifact needs reconciliation; automatic retraining is blocked.')
            return original_hybrid(*args,**kwargs)
        training.train_hybrid_model=guarded_hybrid


def task_worker(job_path):
    job=read_json(job_path)
    if not isinstance(job,dict): raise ValueError('Invalid worker job JSON.')
    os.environ.update(worker_env(job['project']))
    sys.path.insert(0,job['runtime'])
    faulthandler.enable(all_threads=True)
    periodic_traceback = should_enable_periodic_traceback(sys.platform)
    if periodic_traceback:
        faulthandler.dump_traceback_later(600, repeat=True)
    try:
        stage(job,'worker:started',pid=os.getpid(),load_order=job.get('load_order'))
        objects=preload_for_training(job)
        stage(job,'runtime:training_import')
        install_training_hooks(job,objects)
        stage(job,'task:deserialize')
        from joblib.externals import cloudpickle
        with (Path(job['directory'])/'call.pkl').open('rb') as f:
            func=cloudpickle.load(f)
        stage(job,'task:execute')
        result=func()
        stage(job,'task:return')
        atomic_json(Path(job['directory'])/'response.json',{'ok':True,'result':result})
        return 0
    except Exception as exc:
        tb=traceback.format_exc()
        print(tb,file=sys.stderr,flush=True)
        atomic_json(Path(job['directory'])/'response.json',
                    {'ok':False,'error_type':type(exc).__name__,'error':str(exc),'traceback':tb})
        return 1
    finally:
        if periodic_traceback:
            faulthandler.cancel_dump_traceback_later()


class WorkerFailure(RuntimeError):
    pass


class Dispatch:
    def __init__(self,project,runtime,session,native_retries=1,timeout=0):
        self.project=Path(project); self.runtime=Path(runtime); self.session=Path(session)
        self.native_retries=native_retries; self.timeout=timeout
        self.vram_gb=detect_vram_gb()
        self.events=[]; self.failures=[]; self.validation_cache={}
    def event(self,**data):
        data.update(time=now())
        self.events.append(data)
        with (self.session/'events.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(data,default=str)+'\n')
    def inspect(self,key):
        return inspect_training_bundle(self.project,key)
    def dispatch_job(self,key,func,payload):
        from joblib.externals import cloudpickle
        safe=re.sub(r'[^A-Za-z0-9_.-]+','_',key)
        directory=self.session/'tasks'/(safe+'_'+uuid.uuid4().hex[:8])
        directory.mkdir(parents=True)
        with (directory/'call.pkl').open('wb') as f: cloudpickle.dump(func,f)
        profile_index=initial_profile_index(key, self.vram_gb); native_tries=0; serial=0
        if profile_index:
            print('[LOW-VRAM POLICY] Starting directly with original PEFT profile:', key, flush=True)
            self.event(task=key,event='low_vram_profile_start',profile_index=profile_index,vram_gb=self.vram_gb)
        try:
            while True:
                serial+=1
                job={'project':str(self.project),'runtime':str(self.runtime),'task_key':key,
                     'payload':payload or {},'directory':str(directory),'profile_index':profile_index,
                     'load_order':'model_first' if native_tries else 'tokenizer_first'}
                atomic_json(directory/'job.json',job)
                for n in ('response.json','stage.json'):
                    p=directory/n
                    if p.exists(): p.unlink()
                log=directory/('attempt_%02d.log'%serial)
                self.event(task=key,event='worker_launch',profile_index=profile_index,log=str(log))
                code=run_logged([sys.executable,'-u','-X','faulthandler',str(SELF),
                                 '--worker',str(directory/'job.json')],self.project,log,
                                env=worker_env(self.project),timeout=self.timeout)
                response=read_json(directory/'response.json',{}) or {}
                phase=read_json(directory/'stage.json',{}) or {}
                atomic_json(directory/('attempt_%02d_status.json'%serial),
                            {'returncode':code,'exit_hex':exit_hex(code),'response':response,'stage':phase})
                if code==0 and response.get('ok'):
                    return response.get('result') or {}
                if training_root(self.project,key) is not None:
                    recovered=inspect_training_bundle(self.project,key)
                    if recovered.get('complete'):
                        self.event(task=key,event='outputs_verified_after_worker_exit',returncode=code,exit_hex=exit_hex(code))
                        return recovered['payload']
                error=response.get('error') or ('Worker exited '+str(code)+' ('+exit_hex(code)+')')
                self.event(task=key,event='worker_failed',returncode=code,exit_hex=exit_hex(code),
                           phase=phase.get('phase'),error=error)
                if phase.get('had_oom') and profile_index+1<int(phase.get('profile_count',1)):
                    profile_index+=1; native_tries=0
                    print('[OOM] Next ORIGINAL profile in a fresh process:',key,profile_index,flush=True)
                    continue
                # An abrupt process exit is not a normal Python model error.
                if not response and code!=124 and native_tries<self.native_retries:
                    native_tries+=1
                    print('[NATIVE EXIT] One fresh-process retry with changed load order:',key,flush=True)
                    continue
                raise WorkerFailure(error+' | phase='+str(phase.get('phase'))+' | log='+str(log))
        finally:
            # Do not leave dataset-containing serialized closures in the log archive.
            p=directory/'call.pkl'
            if p.exists(): p.unlink()
    def __call__(self,state_db,task_key,func,payload=None,skip_complete=True):
        root=training_root(self.project,task_key)
        row=state_db.get(task_key) or {}
        if root is not None:
            bundle=self.inspect(task_key)
            if skip_complete and bundle.get('complete'):
                result=bundle['payload']
                if not row:
                    state_db.mark_running(task_key,payload or {})
                state_db.mark_complete(task_key,artifact_path=bundle['artifact'],payload=result,
                                       checkpoint_path=result.get('resource',{}).get('best_model_checkpoint'))
                print('[SKIP VERIFIED]',task_key,flush=True)
                self.event(task=task_key,event='skip_verified')
                return True
            if bundle.get('saved_model'):
                print('[RECONCILE ONLY, NO TRAINING]',task_key,flush=True)
                if not bundle.get('repairable'):
                    exc=WorkerFailure('Saved model exists; reconciliation requires review. '+str(bundle['issues']))
                    state_db.mark_running(task_key,payload or {})
                    state_db.mark_failed(task_key,exc,str(exc),retryable=False,payload=payload or {})
                    self.failures.append(task_key); self.event(task=task_key,event='needs_review',error=str(exc))
                    return False
        elif skip_complete and state_db.is_complete(task_key):
            artifact=row.get('artifact_path')
            if artifact:
                p=absolute_path(self.project,artifact)
                if p.exists() and (p.is_file() and p.stat().st_size>0 or p.is_dir() and any(p.iterdir())):
                    print('[SKIP]',task_key,flush=True)
                    self.event(task=task_key,event='skip_existing_nontraining')
                    return True
        state_db.mark_running(task_key,payload or {})
        self.event(task=task_key,event='running')
        try:
            result=self.dispatch_job(task_key,func,payload or {})
            artifact=result.get('artifact_path') if isinstance(result,dict) else None
            if root is not None:
                verified=self.inspect(task_key)
                if not verified.get('complete'):
                    raise WorkerFailure('Process returned but outputs are incomplete: '+str(verified['issues']))
                result=verified['payload']; artifact=verified['artifact']
            elif artifact:
                artifact=str(absolute_path(self.project,artifact))
                if not Path(artifact).exists(): raise WorkerFailure('Reported artifact does not exist: '+artifact)
            if not isinstance(result,dict):
                raise WorkerFailure('Task did not return the original dictionary result contract.')
            checkpoint=result.get('checkpoint_path') or result.get('resource',{}).get('best_model_checkpoint')
            if checkpoint: checkpoint=str(absolute_path(self.project,checkpoint))
            state_db.mark_complete(task_key,artifact_path=artifact,
                payload=result.get('payload',result),checkpoint_path=checkpoint)
            print('[COMPLETE VERIFIED]',task_key,flush=True)
            self.event(task=task_key,event='complete_verified',artifact=artifact)
            return True
        except Exception as exc:
            tb=traceback.format_exc()
            state_db.mark_failed(task_key,exc,tb,retryable=False,payload=payload or {})
            self.failures.append(task_key)
            self.event(task=task_key,event='failed',error=str(exc))
            print('[FAILED; CONTINUING]',task_key,'|',exc,flush=True)
            return False






def audit(project):
    project=Path(project)
    db=project/'results/state.db'
    rows=[]
    if db.is_file():
        c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=20)
        try:
            c.row_factory=sqlite3.Row
            rows=[dict(x) for x in c.execute('SELECT * FROM tasks ORDER BY task_key')]
        finally: c.close()
    checks=[]
    for row in rows:
        key=row['task_key']
        if training_root(project,key) is not None:
            check=inspect_training_bundle(project,key)
            status=('VERIFIED_OUTPUTS' if check['complete'] else
                    'SAVED_MODEL_NEEDS_OUTPUT_REPAIR' if check['repairable'] else
                    'SAVED_MODEL_NEEDS_REVIEW' if check['saved_model'] else 'NOT_COMPLETE')
            checks.append({'task_key':key,'database_status':row['status'],'audit_status':status,
                           'issues':check['issues'],'artifact':check.get('artifact')})
    # Standalone jobs may not have a DB row at all.
    known={r['task_key'] for r in rows}
    models=project/'results/models'
    for selected in sorted(models.glob('*/seed_*/*/selected_result.json')) if models.exists() else []:
        root=selected.parent; protocol=root.parent.parent.name; seed=root.parent.name[5:]
        # Only single-model artifacts can be identified safely without guessing the registry.
        obj=read_json(selected,{}) or {}
        art=absolute_path(project,obj.get('artifact_path',str(root/'full_primary/artifact')))
        meta=read_json(art/'artifact_meta.json',{}) or {}
        if meta.get('type')!='classifier': continue
        key=f'{protocol}:seed{seed}:single:{root.name}'
        if key not in known:
            ck=inspect_training_bundle(project,key)
            checks.append({'task_key':key,'database_status':'no_row',
                           'audit_status':'VERIFIED_OUTPUTS' if ck['complete'] else 'SAVED_MODEL_NEEDS_OUTPUT_REPAIR',
                           'issues':ck['issues'],'artifact':ck.get('artifact')})
    return {'time':now(),'project':str(project.resolve()),'checks':checks,
            'note':'Read-only audit of model outputs; does not prove training protocol equivalence or rerun any model.'}

def _load_supervisor(project):
    """The supervisor is part of this file (formerly FRD_Run_All_Safe.py v1.1.0)."""
    module = sys.modules.get(__name__)
    if module is None or not hasattr(module, 'Dispatch'):
        import types
        module = types.SimpleNamespace(**globals())
    return module


# =============================================================================
# 2. No-large study (formerly FRD_Run_NoLarge_Study.py v1.0.0)
# =============================================================================
# -*- coding: utf-8 -*-


NOLARGE_RUNNER_VERSION = "1.0.0"
NOLARGE_LARGE_KEYS = {"deberta_v3_large", "modernbert_large", "electra_large", "roberta_large"}
SUMMARY_METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "specificity", "balanced_accuracy", "mcc", "tn", "fp", "fn", "tp",
)


def selected_nonlarge_model_keys(registry):
    """Canonical single-model keys whose registry entry is not marked large."""
    return [key for key, spec in registry.items() if not bool(getattr(spec, "large", False))]


def ensemble_candidate_keys(model_registry, hybrid_registry):
    """Models eligible for 3-way ensembles; sentiment-only is a baseline, not a member."""
    return selected_nonlarge_model_keys(model_registry) + list(hybrid_registry.keys())


def planned_training_tasks(model_registry, hybrid_registry, seed=42):
    """Human-auditable training task plan. By construction it contains no large model."""
    singles = selected_nonlarge_model_keys(model_registry)
    tasks = []
    for protocol in ("frd", "opspam"):
        tasks.extend(f"{protocol}:seed{seed}:single:{k}" for k in singles)
        tasks.append(f"{protocol}:seed{seed}:sentiment_only")
        tasks.extend(f"{protocol}:seed{seed}:hybrid:{k}" for k in hybrid_registry)
    return tasks


def _nl_rank_auc(y_true, prob):
    """Binary ROC-AUC via rank sum, dependency-free for selection/test helpers."""
    import numpy as np
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum = float(ranks[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _nl_simple_metrics(y_true, prob, threshold=0.5):
    import numpy as np
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-15, precision + recall)
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _nl_rank_auc(y, p),
    }


def nl_rank_equal_soft_triples(y_true, probs, top_n=3):
    """Rank 3-model equal-soft ensembles using validation data only.

    No test-set argument exists by design. This makes accidental test-based
    ensemble selection structurally impossible in this function.
    """
    import numpy as np
    names = sorted(probs)
    if len(names) < 3:
        raise ValueError("At least three candidate models are required for triple ensemble search.")
    rows = []
    for members in combinations(names, 3):
        p = np.mean(np.column_stack([np.asarray(probs[m], dtype=float) for m in members]), axis=1)
        m = _nl_simple_metrics(y_true, p, 0.5)
        rows.append({
            "members": tuple(members),
            "validation_accuracy": float(m["accuracy"]),
            "validation_f1": float(m["f1"]),
            "validation_roc_auc": float(m["roc_auc"]),
        })
    rows.sort(
        key=lambda r: (
            r["validation_accuracy"], r["validation_f1"],
            r["validation_roc_auc"], tuple(r["members"]),
        ),
        reverse=True,
    )
    if top_n is None:
        return rows
    return rows[: max(0, int(top_n))]




def _nl_read_split(project, protocol):
    import pandas as pd
    name = "frd_split.csv" if protocol == "frd" else "opspam_split.csv"
    path = Path(project) / "results" / "splits" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Persisted {protocol} split is required; this runner never creates a new split: {path}"
        )
    frame = pd.read_csv(path)
    text_col = "text_" if protocol == "frd" else "text"
    required = {text_col, "label_num", "split"}
    id_field = "sample_id" if "sample_id" in frame.columns else "source_index" if "source_index" in frame.columns else None
    if not required.issubset(frame.columns) or id_field is None:
        raise RuntimeError(f"Incompatible persisted {protocol} split: {path}")
    if set(frame["split"].astype(str).unique()) != {"train", "validation", "test"}:
        raise RuntimeError(f"{protocol} split must contain train/validation/test")
    if frame[id_field].isna().any() or frame[id_field].duplicated().any():
        raise RuntimeError(f"{protocol} split IDs are missing or duplicated")
    return frame, text_col, path


def _nl_build_features(runtime, results_dir, protocol, frame, text_col):
    import joblib
    from frdexp.features import build_feature_cache, fit_feature_scaler
    cache = Path(results_dir) / "feature_cache" / f"{protocol}_sentiment.csv"
    features = build_feature_cache(frame, text_col, cache)
    scaler_path = Path(results_dir) / "feature_cache" / f"{protocol}_sentiment_scaler.joblib"
    scaler = fit_feature_scaler(features.loc[frame.split == "train"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return features, scaler


def _training_bundle_complete(orchestrator, results_dir, protocol, seed, key):
    root = Path(results_dir) / "models" / protocol / f"seed_{seed}" / key
    return orchestrator.load_prediction_bundle(root) is not None


def _run_protocol_training(dispatch, state, results_dir, protocol, frame, text_col,
                           features, scaler, model_specs, hybrid_specs, config, seed):
    from frdexp.features import FEATURE_NAMES
    from frdexp.orchestrator import run_sentiment_baseline
    from frdexp.training import train_single_model, train_hybrid_model

    singles = selected_nonlarge_model_keys(model_specs)
    for key in singles:
        spec = model_specs[key]
        task = f"{protocol}:seed{seed}:single:{key}"
        root = Path(results_dir) / "models" / protocol / f"seed_{seed}" / key
        print(f"\n=== {task} ===", flush=True)
        dispatch(
            state, task,
            lambda spec=spec, root=root: train_single_model(spec, frame, text_col, seed, config, root),
            payload={"protocol": protocol, "model": key, "seed": seed, "hf_id": spec.hf_id,
                     "scope": "no_large_study"},
        )

    s_task = f"{protocol}:seed{seed}:sentiment_only"
    s_root = Path(results_dir) / "models" / protocol / f"seed_{seed}" / "sentiment_only" / "cpu_lr"
    print(f"\n=== {s_task} ===", flush=True)
    dispatch(
        state, s_task,
        lambda: run_sentiment_baseline(frame, features, seed, s_root),
        payload={"protocol": protocol, "model": "sentiment_only", "seed": seed,
                 "scope": "no_large_study"},
    )

    for key, hspec in hybrid_specs.items():
        enc = model_specs[hspec["encoder"]]
        task = f"{protocol}:seed{seed}:hybrid:{key}"
        root = Path(results_dir) / "models" / protocol / f"seed_{seed}" / key
        print(f"\n=== {task} ===", flush=True)
        dispatch(
            state, task,
            lambda key=key, hspec=hspec, enc=enc, root=root: train_hybrid_model(
                key, hspec, enc, frame, text_col, features, list(FEATURE_NAMES),
                scaler, seed, config, root
            ),
            payload={"protocol": protocol, "model": key, "seed": seed,
                     "encoder": enc.hf_id, "scope": "no_large_study"},
        )


def _collect_available_candidates(results_dir, protocol, seed, candidate_keys):
    from frdexp.orchestrator import load_prediction_bundle
    base = Path(results_dir) / "models" / protocol / f"seed_{seed}"
    return [k for k in candidate_keys if load_prediction_bundle(base / k) is not None]


def nl_build_selected_triple_ensembles(results_dir, protocol, seed, candidate_keys, out_root, top_n=3):
    """Validation-only 3-model search; test set is touched only for selected triples."""
    import pandas as pd
    from frdexp.orchestrator import load_prediction_bundle, _align_probabilities, _save_ensemble_method
    from frdexp.ensembles import weighted_probability, optimize_soft_weights

    model_base = Path(results_dir) / "models" / protocol / f"seed_{seed}"
    bundles = {k: load_prediction_bundle(model_base / k) for k in candidate_keys}
    bundles = {k: v for k, v in bundles.items() if v is not None}
    if len(bundles) < 3:
        raise RuntimeError(f"Need at least three complete {protocol} candidates; got {sorted(bundles)}")
    val_base, val_probs = _align_probabilities(bundles, "validation")

    # Search never receives test probabilities.
    all_ranked = nl_rank_equal_soft_triples(val_base.y_true.to_numpy(dtype=int), val_probs, top_n=None)
    selected = all_ranked[: min(int(top_n), len(all_ranked))]
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    search_csv = out_root / "triple_search_validation.csv"
    with search_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "member_1", "member_2", "member_3",
                                          "validation_accuracy", "validation_f1", "validation_roc_auc"])
        w.writeheader()
        for i, row in enumerate(all_ranked, 1):
            w.writerow({"rank": i, "member_1": row["members"][0], "member_2": row["members"][1],
                        "member_3": row["members"][2],
                        "validation_accuracy": row["validation_accuracy"],
                        "validation_f1": row["validation_f1"],
                        "validation_roc_auc": row["validation_roc_auc"]})

    # Only now load the held-out test predictions, after the triples are fixed.
    test_base, test_probs = _align_probabilities(bundles, "test")
    created = []
    selection = []
    for rank, row in enumerate(selected, 1):
        members = tuple(row["members"])
        group = f"triple_rank_{rank:02d}__" + "__".join(members)
        vp = {m: val_probs[m] for m in members}
        tp = {m: test_probs[m] for m in members}
        equal = {m: 1.0 / 3.0 for m in members}
        eq_payload = _save_ensemble_method(
            out_root / group / "equal_soft", group, "equal_soft", seed,
            val_base, test_base,
            weighted_probability(vp, equal), weighted_probability(tp, equal),
            {"members": members, "weights": equal, "protocol": protocol,
             "selection_data": "validation_only", "triple_rank": rank},
        )
        weights = optimize_soft_weights(val_base.y_true.to_numpy(dtype=int), vp, step=0.05)
        wt_payload = _save_ensemble_method(
            out_root / group / "weighted_soft", group, "weighted_soft", seed,
            val_base, test_base,
            weighted_probability(vp, weights), weighted_probability(tp, weights),
            {"members": members, "weights": weights, "protocol": protocol,
             "selection_data": "validation_only", "triple_rank": rank,
             "weight_search_step": 0.05},
        )
        created.extend([eq_payload, wt_payload])
        selection.append({"rank": rank, "members": members,
                          "validation_search": row,
                          "equal_soft_dir": str((out_root / group / "equal_soft").resolve()),
                          "weighted_soft_dir": str((out_root / group / "weighted_soft").resolve())})

    manifest = {
        "runner_version": NOLARGE_RUNNER_VERSION,
        "protocol": protocol,
        "seed": seed,
        "selection_data": "validation_only",
        "candidate_pool": sorted(bundles),
        "combination_size": 3,
        "total_combinations": len(all_ranked),
        "selected_top_n": len(selection),
        "selected": selection,
        "search_csv": str(search_csv.resolve()),
    }
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {"artifact_path": str((out_root / "selection_manifest.json").resolve()),
            "payload": {"protocol": protocol, "selected": selection,
                        "total_combinations": len(all_ranked), "created_methods": len(created)}}


def run_sentiment_zero_shot(results_dir, seed, opspam_frame, opspam_features, out_dir):
    """Apply the FRD-trained polarity-only LR baseline to the full OpSpam corpus."""
    import joblib
    import numpy as np
    from frdexp.metrics import compute_metrics
    from frdexp.prediction import save_predictions
    from frdexp.orchestrator import load_selected_result

    root = Path(results_dir) / "models" / "frd" / f"seed_{seed}" / "sentiment_only"
    selected = load_selected_result(root)
    if not selected:
        raise FileNotFoundError("FRD sentiment-only selected_result.json is missing")
    artifact = Path(selected["artifact_path"])
    model = joblib.load(artifact)
    X = np.asarray(opspam_features[["polarity"]], dtype=float)
    prob = model.predict_proba(X)[:, 1]
    threshold = float(selected.get("selected_threshold", 0.5))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "zero_shot_frd_to_opspam",
        "model": "sentiment_only",
        "seed": seed,
        "source_threshold": threshold,
        "default": compute_metrics(opspam_frame.label_num, prob, 0.5),
        "frd_validation_calibrated": compute_metrics(opspam_frame.label_num, prob, threshold),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_predictions(out_dir / "predictions.csv", opspam_frame, prob, threshold,
                     "sentiment_only", seed, None)
    return {"artifact_path": str((out_dir / "predictions.csv").resolve()), "payload": payload}


def nl_build_external_triple_ensembles(results_dir, seed, frd_ensemble_root, zero_shot_root, out_root):
    """Transfer FRD-selected triples to OpSpam without using OpSpam labels for selection."""
    import numpy as np
    import pandas as pd
    from frdexp.metrics import compute_metrics
    from frdexp.prediction import save_predictions

    frd_ensemble_root = Path(frd_ensemble_root)
    zero_shot_root = Path(zero_shot_root)
    out_root = Path(out_root)
    manifest = json.loads((frd_ensemble_root / "selection_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for selected in manifest["selected"]:
        rank = int(selected["rank"])
        members = tuple(selected["members"])
        pred_frames = {}
        base = None
        for m in members:
            d = pd.read_csv(zero_shot_root / m / "predictions.csv").sort_values("sample_id").reset_index(drop=True)
            if base is None:
                base = d
            elif not np.array_equal(base.sample_id.to_numpy(), d.sample_id.to_numpy()) or not np.array_equal(base.y_true.to_numpy(), d.y_true.to_numpy()):
                raise RuntimeError(f"Zero-shot prediction alignment mismatch for {m}")
            pred_frames[m] = d.y_prob_cg.to_numpy(float)
        for method_name, source_dir_key in (("equal_soft", "equal_soft_dir"), ("weighted_soft", "weighted_soft_dir")):
            source_dir = Path(selected[source_dir_key])
            meta = json.loads((source_dir / "ensemble_meta.json").read_text(encoding="utf-8"))
            weights = {k: float(v) for k, v in meta["weights"].items()}
            wsum = sum(weights.values())
            prob = sum(pred_frames[k] * weights[k] for k in members) / wsum
            threshold = float(meta["selected_threshold"])
            group = f"triple_rank_{rank:02d}__" + "__".join(members)
            target = out_root / group / method_name
            target.mkdir(parents=True, exist_ok=True)
            payload = {
                "protocol": "zero_shot_frd_to_opspam",
                "model": group,
                "method": method_name,
                "seed": seed,
                "members": members,
                "weights": weights,
                "source_threshold": threshold,
                "selection_data": "FRD_validation_only",
                "default": compute_metrics(base.y_true, prob, 0.5),
                "frd_validation_calibrated": compute_metrics(base.y_true, prob, threshold),
            }
            (target / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            frame = pd.DataFrame({"source_index": base.sample_id, "label_num": base.y_true})
            save_predictions(target / "predictions.csv", frame, prob, threshold, group, seed, None)
            rows.append(payload)
    summary = out_root / "external_triple_ensemble_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"artifact_path": str(summary.resolve()), "payload": {"created": len(rows)}}


def run_no_large_statistics(results_dir, seed, candidate_keys, ensemble_root, out_root,
                            bootstrap_seed=314159):
    import numpy as np
    import pandas as pd
    from frdexp.orchestrator import load_prediction_bundle
    from frdexp.statistics import bootstrap_ci, mcnemar_test, paired_bootstrap_difference, holm_adjust

    results_dir = Path(results_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_base = results_dir / "models" / "frd" / f"seed_{seed}"
    candidates = {}
    for key in list(candidate_keys) + ["sentiment_only"]:
        b = load_prediction_bundle(model_base / key)
        if b:
            candidates[key] = b

    ensemble_keys = []
    for mp in Path(ensemble_root).glob("*/*/metrics.json"):
        vp = mp.parent / "validation_predictions.csv"
        tp = mp.parent / "test_predictions.csv"
        if not (vp.exists() and tp.exists()):
            continue
        d = json.loads(mp.read_text(encoding="utf-8"))
        key = f"ensemble::{mp.parent.parent.name}::{mp.parent.name}"
        candidates[key] = {"metrics": d, "validation": pd.read_csv(vp),
                           "test": pd.read_csv(tp), "result_dir": mp.parent}
        ensemble_keys.append(key)

    ci_rows = []
    for key, bundle in candidates.items():
        t = bundle["test"]
        threshold = float(bundle["metrics"].get("selected_threshold", 0.5))
        for metric in ("accuracy", "f1", "roc_auc"):
            ci = bootstrap_ci(t.y_true, t.y_prob_cg, metric, threshold,
                              n_boot=1000, seed=bootstrap_seed)
            ci_rows.append({"model": key, "metric": metric, **ci})
    pd.DataFrame(ci_rows).to_csv(out_root / "confidence_intervals.csv", index=False)

    base_models = [k for k in candidate_keys if k in candidates]
    def vscore(key):
        m = candidates[key]["metrics"]["validation_default"]
        return (m["accuracy"], m["f1"], m["roc_auc"], key)
    best_model = max(base_models, key=vscore) if base_models else None
    best_ensemble = max(ensemble_keys, key=vscore) if ensemble_keys else None

    comparisons = [
        ("bert", "bert_polarity"),
        ("bert", "bert_sentiment_profile"),
        ("deberta_v3_base", "deberta_v3_sentiment_gated"),
    ]
    if best_model and best_model != "bert" and "bert" in candidates:
        comparisons.append((best_model, "bert"))
    if best_model and best_ensemble:
        comparisons.append((best_ensemble, best_model))

    rows = []
    pvals = []
    for a, b in comparisons:
        if a not in candidates or b not in candidates:
            continue
        da = candidates[a]["test"].sort_values("sample_id").reset_index(drop=True)
        db = candidates[b]["test"].sort_values("sample_id").reset_index(drop=True)
        if not np.array_equal(da.sample_id.to_numpy(), db.sample_id.to_numpy()):
            continue
        th_a = float(candidates[a]["metrics"].get("selected_threshold", 0.5))
        th_b = float(candidates[b]["metrics"].get("selected_threshold", 0.5))
        mc = mcnemar_test(da.y_true, (da.y_prob_cg >= th_a).astype(int),
                          (db.y_prob_cg >= th_b).astype(int))
        pvals.append(mc["p"])
        row = {"model_a": a, "model_b": b, **mc}
        for metric in ("accuracy", "f1", "roc_auc"):
            d = paired_bootstrap_difference(
                da.y_true, da.y_prob_cg, db.y_prob_cg, metric,
                th_a, th_b, n_boot=1000, seed=bootstrap_seed,
            )
            row[f"{metric}_diff"] = d["difference"]
            row[f"{metric}_ci_low"] = d["low"]
            row[f"{metric}_ci_high"] = d["high"]
        rows.append(row)
    if rows:
        adjusted = holm_adjust(pvals)
        for row, p in zip(rows, adjusted):
            row["holm_p"] = p
            row["significant_0_05"] = bool(p < 0.05)
    pd.DataFrame(rows).to_csv(out_root / "statistical_tests.csv", index=False)

    ablations = []
    for baseline, augmented in (("bert", "bert_polarity"),
                                ("bert", "bert_sentiment_profile"),
                                ("deberta_v3_base", "deberta_v3_sentiment_gated")):
        if baseline not in candidates or augmented not in candidates:
            continue
        a = candidates[baseline]["metrics"]["test_calibrated"]
        b = candidates[augmented]["metrics"]["test_calibrated"]
        row = {"baseline": baseline, "augmented": augmented}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"):
            row[f"{metric}_baseline"] = a[metric]
            row[f"{metric}_augmented"] = b[metric]
            row[f"{metric}_delta"] = b[metric] - a[metric]
        ablations.append(row)
    pd.DataFrame(ablations).to_csv(out_root / "ablation_results.csv", index=False)

    manifest = {
        "seed": seed,
        "scope": "no_large_only",
        "candidate_models": sorted(k for k in candidates if not k.startswith("ensemble::")),
        "candidate_ensembles": sorted(ensemble_keys),
        "best_model_validation": best_model,
        "best_ensemble_validation": best_ensemble,
        "comparisons": len(rows),
    }
    path = out_root / "statistics_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"artifact_path": str(path.resolve()), "payload": manifest}


def _flatten_metrics(prefix, metrics):
    out = {}
    for m in SUMMARY_METRICS:
        out[f"{prefix}_{m}"] = (metrics or {}).get(m)
    return out


def _candidate_summary(results_dir, protocol, seed, candidate_keys, out_csv):
    from frdexp.orchestrator import load_prediction_bundle
    rows = []
    base = Path(results_dir) / "models" / protocol / f"seed_{seed}"
    for key in list(candidate_keys) + ["sentiment_only"]:
        b = load_prediction_bundle(base / key)
        if not b:
            continue
        metrics = b["metrics"]
        row = {"model": key, "selected_threshold": metrics.get("selected_threshold")}
        row.update(_flatten_metrics("test_default", metrics.get("test_default")))
        row.update(_flatten_metrics("test_calibrated", metrics.get("test_calibrated")))
        rows.append(row)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else ["model"]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def _nl_zero_shot_summary(results_dir, seed, candidate_keys, out_csv):
    rows = []
    root = Path(results_dir) / "external_validation" / f"seed_{seed}" / "zero_shot"
    for key in list(candidate_keys) + ["sentiment_only"]:
        p = root / key / "metrics.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        row = {"model": key, "source_threshold": d.get("source_threshold")}
        row.update(_flatten_metrics("default", d.get("default")))
        row.update(_flatten_metrics("frd_calibrated", d.get("frd_validation_calibrated")))
        rows.append(row)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else ["model"]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return rows


def _write_markdown_report(path, frd_rows, opspam_rows, zero_rows, ensemble_manifest, stats_manifest):
    def pct(v):
        return "" if v is None else f"{100*float(v):.2f}%"
    lines = [
        "# No-Large FRD Study Summary",
        "",
        "Large models were explicitly excluded from this run and from ensemble selection.",
        "All 3-model ensemble combinations were ranked on validation predictions only; only the top three triples were evaluated on the held-out test partition.",
        "",
        "## FRD models",
        "",
        "| Model | Accuracy | F1 | ROC-AUC | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sorted(frd_rows, key=lambda x: (x.get("test_default_accuracy") or -1), reverse=True):
        lines.append(f"| {r['model']} | {pct(r.get('test_default_accuracy'))} | {pct(r.get('test_default_f1'))} | {r.get('test_default_roc_auc','')} | {r.get('test_default_mcc','')} |")
    lines += ["", "## Selected FRD 3-model ensembles", ""]
    for item in ensemble_manifest.get("selected", []):
        lines.append(f"- Rank {item['rank']}: {', '.join(item['members'])}")
    lines += ["", "## OpSpam fine-tuned models", "",
              "| Model | Accuracy | F1 | ROC-AUC |", "|---|---:|---:|---:|"]
    for r in sorted(opspam_rows, key=lambda x: (x.get("test_default_accuracy") or -1), reverse=True):
        lines.append(f"| {r['model']} | {pct(r.get('test_default_accuracy'))} | {pct(r.get('test_default_f1'))} | {r.get('test_default_roc_auc','')} |")
    lines += ["", "## FRD -> OpSpam zero-shot", "",
              "| Model | Accuracy | F1 | ROC-AUC |", "|---|---:|---:|---:|"]
    for r in sorted(zero_rows, key=lambda x: (x.get("default_accuracy") or -1), reverse=True):
        lines.append(f"| {r['model']} | {pct(r.get('default_accuracy'))} | {pct(r.get('default_f1'))} | {r.get('default_roc_auc','')} |")
    lines += ["", "## Statistical selection", "",
              f"- Best non-large model by validation: `{stats_manifest.get('best_model_validation')}`",
              f"- Best selected ensemble by validation: `{stats_manifest.get('best_ensemble_validation')}`",
              "- Detailed confidence intervals, McNemar/Holm results, paired bootstrap differences, and ablations are in the statistics subdirectory.",
              ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def nolarge_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-retries", type=int, default=1, choices=(0, 1))
    parser.add_argument("--max-task-hours", type=float, default=0)
    args = parser.parse_args(argv)
    if args.max_task_hours < 0:
        parser.error("--max-task-hours must be nonnegative")

    project = args.project.expanduser().resolve()
    supervisor = _load_supervisor(project)
    frd, frd_text, frd_split = _nl_read_split(project, "frd")
    opspam, opspam_text, opspam_split = _nl_read_split(project, "opspam")

    print(f"FRD no-large study runner {NOLARGE_RUNNER_VERSION}", flush=True)
    print("PROJECT:", project, flush=True)
    print("SUPERVISOR:", supervisor.VERSION, flush=True)
    print("SCOPE: no-large singles + hybrids + sentiment baseline + OpSpam + top-3 triple ensembles + statistics", flush=True)
    print("EXCLUDED LARGE MODELS:", ", ".join(sorted(NOLARGE_LARGE_KEYS)), flush=True)
    print("FRD SPLIT:", frd_split, flush=True)
    print("OPSPAM SPLIT:", opspam_split, flush=True)

    with supervisor.ProjectLock(project / ".frd_supervisor" / "runner.lock"):
        runtime, _ = supervisor.original_runtime(project)
        session = project / ".frd_supervisor" / "sessions" / ("nolarge_study_" + supervisor.stamp())
        session.mkdir(parents=True, exist_ok=True)
        results_dir = project / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        supervisor.backup_database(results_dir / "state.db", session / "state_before.sqlite3")
        os.environ.update(supervisor.worker_env(project))
        sys.path.insert(0, str(runtime))

        from config import CONFIG
        from frdexp.model_registry import MODEL_SPECS, HYBRID_SPECS
        from frdexp.state import StateDB
        from frdexp.orchestrator import load_prediction_bundle, run_zero_shot_model

        nonlarge = selected_nonlarge_model_keys(MODEL_SPECS)
        if set(nonlarge) & NOLARGE_LARGE_KEYS or any(bool(getattr(MODEL_SPECS[k], "large", False)) for k in nonlarge):
            raise RuntimeError("Large-model scope leak detected; refusing to start.")
        candidates = ensemble_candidate_keys(MODEL_SPECS, HYBRID_SPECS)
        if any(k in NOLARGE_LARGE_KEYS for k in candidates):
            raise RuntimeError("Large model leaked into ensemble candidate pool; refusing to start.")
        print("NON-LARGE SINGLE MODELS:", ", ".join(nonlarge), flush=True)
        print("HYBRIDS:", ", ".join(HYBRID_SPECS), flush=True)
        print("TRIPLE ENSEMBLE CANDIDATES:", ", ".join(candidates), flush=True)

        frd_features, frd_scaler = _nl_build_features(runtime, results_dir, "frd", frd, frd_text)
        opspam_features, opspam_scaler = _nl_build_features(runtime, results_dir, "opspam", opspam, opspam_text)

        dispatch = supervisor.Dispatch(project, runtime, session,
                                       native_retries=args.native_retries,
                                       timeout=args.max_task_hours * 3600)
        state = StateDB(results_dir / "state.db")
        scoped_root = results_dir / "no_large_study" / f"seed_{args.seed}"
        scoped_root.mkdir(parents=True, exist_ok=True)
        try:
            print("\n===== FRD NON-LARGE TRAINING / VERIFICATION =====", flush=True)
            _run_protocol_training(dispatch, state, results_dir, "frd", frd, frd_text,
                                   frd_features, frd_scaler, MODEL_SPECS, HYBRID_SPECS,
                                   CONFIG, args.seed)

            available_frd = _collect_available_candidates(results_dir, "frd", args.seed, candidates)
            print("\nFRD ensemble-ready candidates:", ", ".join(available_frd), flush=True)
            ens_task = f"nolarge:seed{args.seed}:frd_triple_ensemble_search"
            dispatch(
                state, ens_task,
                lambda: nl_build_selected_triple_ensembles(
                    results_dir, "frd", args.seed, available_frd,
                    scoped_root / "ensembles" / "frd", top_n=3),
                payload={"seed": args.seed, "protocol": "frd", "combination_size": 3,
                         "selection_data": "validation_only", "large_models": "excluded"},
                skip_complete=False,
            )

            print("\n===== FRD -> OPSPAM ZERO-SHOT =====", flush=True)
            zero_root = results_dir / "external_validation" / f"seed_{args.seed}" / "zero_shot"
            for key in available_frd:
                root = results_dir / "models" / "frd" / f"seed_{args.seed}" / key
                out = zero_root / key
                task = f"opspam_zero_shot:seed{args.seed}:{key}"
                dispatch(
                    state, task,
                    lambda key=key, root=root, out=out: run_zero_shot_model(
                        key, root, opspam, opspam_features, CONFIG, args.seed, out),
                    payload={"protocol": "zero_shot_frd_to_opspam", "model": key,
                             "seed": args.seed, "scope": "no_large_study"},
                )
            dispatch(
                state, f"opspam_zero_shot:seed{args.seed}:sentiment_only",
                lambda: run_sentiment_zero_shot(results_dir, args.seed, opspam,
                                                opspam_features, zero_root / "sentiment_only"),
                payload={"protocol": "zero_shot_frd_to_opspam", "model": "sentiment_only",
                         "seed": args.seed, "scope": "no_large_study"},
            )
            dispatch(
                state, f"nolarge:seed{args.seed}:external_triple_ensembles",
                lambda: nl_build_external_triple_ensembles(
                    results_dir, args.seed, scoped_root / "ensembles" / "frd",
                    zero_root, scoped_root / "external_triple_ensembles"),
                payload={"seed": args.seed, "selection_data": "FRD_validation_only",
                         "scope": "no_large_study"},
                skip_complete=False,
            )

            print("\n===== OPSPAM NON-LARGE FINE-TUNING / VERIFICATION =====", flush=True)
            _run_protocol_training(dispatch, state, results_dir, "opspam", opspam, opspam_text,
                                   opspam_features, opspam_scaler, MODEL_SPECS, HYBRID_SPECS,
                                   CONFIG, args.seed)
            available_opspam = _collect_available_candidates(results_dir, "opspam", args.seed, candidates)
            print("\nOpSpam ensemble-ready candidates:", ", ".join(available_opspam), flush=True)
            dispatch(
                state, f"nolarge:seed{args.seed}:opspam_triple_ensemble_search",
                lambda: nl_build_selected_triple_ensembles(
                    results_dir, "opspam", args.seed, available_opspam,
                    scoped_root / "ensembles" / "opspam", top_n=3),
                payload={"seed": args.seed, "protocol": "opspam", "combination_size": 3,
                         "selection_data": "validation_only", "large_models": "excluded"},
                skip_complete=False,
            )

            dispatch(
                state, f"nolarge:seed{args.seed}:statistics",
                lambda: run_no_large_statistics(
                    results_dir, args.seed, available_frd,
                    scoped_root / "ensembles" / "frd",
                    scoped_root / "statistics", CONFIG.bootstrap_seed),
                payload={"seed": args.seed, "scope": "no_large_only"},
                skip_complete=False,
            )
        finally:
            state.close()

        frd_rows = _candidate_summary(results_dir, "frd", args.seed, candidates,
                                      scoped_root / "frd_model_summary.csv")
        opspam_rows = _candidate_summary(results_dir, "opspam", args.seed, candidates,
                                         scoped_root / "opspam_finetune_summary.csv")
        zero_rows = _nl_zero_shot_summary(results_dir, args.seed, candidates,
                                       scoped_root / "opspam_zero_shot_summary.csv")
        ensemble_manifest_path = scoped_root / "ensembles" / "frd" / "selection_manifest.json"
        stats_manifest_path = scoped_root / "statistics" / "statistics_manifest.json"
        ensemble_manifest = json.loads(ensemble_manifest_path.read_text(encoding="utf-8")) if ensemble_manifest_path.exists() else {}
        stats_manifest = json.loads(stats_manifest_path.read_text(encoding="utf-8")) if stats_manifest_path.exists() else {}
        report = scoped_root / "paper_ready_summary.md"
        _write_markdown_report(report, frd_rows, opspam_rows, zero_rows,
                               ensemble_manifest, stats_manifest)

        failed = sorted(set(dispatch.failures))
        summary = {
            "runner_version": NOLARGE_RUNNER_VERSION,
            "status": "COMPLETE" if not failed else "INCOMPLETE",
            "large_models_excluded": sorted(NOLARGE_LARGE_KEYS),
            "nonlarge_single_models": nonlarge,
            "hybrids": list(HYBRID_SPECS),
            "ensemble_candidate_pool": candidates,
            "ensemble_selection": "all 3-model combinations ranked on validation only; top 3 carried to test",
            "failed_tasks": failed,
            "output_root": str(scoped_root.resolve()),
            "paper_ready_summary": str(report.resolve()),
            "session": str(session.resolve()),
        }
        supervisor.atomic_json(scoped_root / "run_summary.json", summary)
        supervisor.atomic_json(session / "no_large_study_summary.json", summary)
        print("\n===== NO-LARGE STUDY SUMMARY =====", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("\nRESULTS:", scoped_root, flush=True)
        return 0 if not failed else 2

# =============================================================================
# 3. Sentiment extension (formerly FRD_Run_Sentiment_Extension.py)
# =============================================================================
# -*- coding: utf-8 -*-


SENTIMENT_RUNNER_VERSION = "1.0.2"
SENTIMENT_LARGE_KEYS = {"deberta_v3_large", "modernbert_large", "electra_large", "roberta_large"}

# New profile variants requested after the original no-large study.  The tuple is
# (encoder model key, gated fusion).  All use the same 9 FEATURE_NAMES features.
NEW_SENTIMENT_VARIANTS = {
    "roberta_sentiment_profile": ("roberta", False),
    "deberta_sentiment_profile": ("deberta", False),
    "modernbert_sentiment_profile": ("modernbert_base", False),
    "electra_sentiment_profile": ("electra_base", False),
    "distilbert_sentiment_profile": ("distilbert", False),
}

# One plain model and one 9-D sentiment-profile model per backbone.  BERT uses the
# existing profile variant; DeBERTa-v3 uses the already-planned gated profile.
BACKBONE_CANDIDATES = {
    "bert": ("bert", "bert_sentiment_profile"),
    "roberta": ("roberta", "roberta_sentiment_profile"),
    "deberta": ("deberta", "deberta_sentiment_profile"),
    "deberta_v3_base": ("deberta_v3_base", "deberta_v3_sentiment_gated"),
    "modernbert_base": ("modernbert_base", "modernbert_sentiment_profile"),
    "electra_base": ("electra_base", "electra_sentiment_profile"),
    "distilbert": ("distilbert", "distilbert_sentiment_profile"),
}


def safe_python_path(project: Path, platform_name: str | None = None) -> Path:
    name = sys.platform if platform_name is None else str(platform_name)
    if name.lower().startswith("win"):
        return Path(project) / ".venv_frd_safe" / "Scripts" / "python.exe"
    return Path(project) / ".venv_frd_safe" / "bin" / "python"


def _same_path(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except Exception:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def ensure_safe_interpreter(project: Path, argv: list[str] | None = None) -> None:
    """Replace a wrong interpreter before importing torch/transformers/pandas."""
    expected = safe_python_path(project)
    if _same_path(Path(sys.executable), expected):
        return
    if not expected.is_file():
        raise RuntimeError(
            "Project-local safe environment is missing: " + str(expected) + "\n"
            "Run: python FRD_Run_Transformer_Study.py setup. No package changes were attempted."
        )
    if os.environ.get("FRD_SENTEXT_SAFE_REEXEC") == "1":
        raise RuntimeError(
            "Safe-environment relaunch was requested but the child interpreter is still wrong.\n"
            f"Expected: {expected}\nActual: {sys.executable}"
        )
    args = [str(expected), str(Path(__file__).resolve())] + list(sys.argv[1:] if argv is None else argv)
    print("[SAFE ENV] Relaunching with:", expected, flush=True)
    if sys.platform.lower().startswith("win"):
        # Windows venv launchers plus paths containing spaces are unreliable with
        # os.execv(): the base interpreter can misinterpret the venv executable
        # itself as the script path.  subprocess handles CreateProcess quoting.
        env = os.environ.copy()
        env["FRD_SENTEXT_SAFE_REEXEC"] = "1"
        completed = subprocess.run(args, env=env)
        raise SystemExit(int(completed.returncode))
    os.environ["FRD_SENTEXT_SAFE_REEXEC"] = "1"
    os.execv(str(expected), args)


def precision_override_for_family(family: str):
    family = str(family)
    if family == "deberta":
        return "fp32"
    if family == "deberta_v3":
        return "fp16"
    return None


def all_triples(names):
    return list(combinations(sorted(names), 3))


def select_backbone_representatives(validation_metrics, backbone_candidates):
    """Pick one candidate per backbone using validation metrics only.

    Ties are deliberately resolved in favor of the first item in each candidate
    tuple (the plain backbone), avoiding gratuitous complexity when validation
    evidence is identical.
    """
    selected = {}
    for backbone, choices in backbone_candidates.items():
        available = [c for c in choices if c in validation_metrics]
        if not available:
            continue
        best = available[0]
        best_score = tuple(float(validation_metrics[best].get(k, float("-inf")))
                           for k in ("accuracy", "f1", "roc_auc"))
        for cand in available[1:]:
            score = tuple(float(validation_metrics[cand].get(k, float("-inf")))
                          for k in ("accuracy", "f1", "roc_auc"))
            if score > best_score:
                best, best_score = cand, score
        selected[backbone] = best
    return selected


def read_json_if_exists(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return default




def _se_read_split(project: Path, protocol: str):
    import pandas as pd
    name = "frd_split.csv" if protocol == "frd" else "opspam_split.csv"
    path = Path(project) / "results" / "splits" / name
    if not path.is_file():
        raise FileNotFoundError(f"Persisted split is required; refusing to create a new split: {path}")
    frame = pd.read_csv(path)
    text_col = "text_" if protocol == "frd" else "text"
    required = {text_col, "label_num", "split"}
    id_field = "sample_id" if "sample_id" in frame.columns else "source_index" if "source_index" in frame.columns else None
    if not required.issubset(frame.columns) or id_field is None:
        raise RuntimeError(f"Incompatible persisted {protocol} split: {path}")
    if set(frame["split"].astype(str).unique()) != {"train", "validation", "test"}:
        raise RuntimeError(f"{protocol} split must contain train/validation/test")
    if frame[id_field].isna().any() or frame[id_field].duplicated().any():
        raise RuntimeError(f"{protocol} split IDs are missing or duplicated")
    return frame, text_col, path


def _se_build_features(results_dir: Path, protocol: str, frame, text_col: str):
    import joblib
    from frdexp.features import build_feature_cache, fit_feature_scaler
    cache = Path(results_dir) / "feature_cache" / f"{protocol}_sentiment.csv"
    features = build_feature_cache(frame, text_col, cache)
    scaler_path = Path(results_dir) / "feature_cache" / f"{protocol}_sentiment_scaler.joblib"
    scaler = fit_feature_scaler(features.loc[frame.split == "train"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return features, scaler


def _forced_precision_call(callable_obj, precision_override=None):
    """Run a runtime training callable with a narrowly-scoped precision override."""
    if not precision_override:
        return callable_obj()
    import frdexp.training as training
    original = training._precision_mode
    training._precision_mode = lambda: str(precision_override)
    try:
        return callable_obj()
    finally:
        training._precision_mode = original


def _single_training_callable(spec, frame, text_col, seed, config, root, precision_override=None):
    def run():
        import frdexp.training as training
        return _forced_precision_call(
            lambda: training.train_single_model(spec, frame, text_col, seed, config, root),
            precision_override,
        )
    return run


def _hybrid_training_callable(key, hspec, enc, frame, text_col, features, feature_names,
                              scaler, seed, config, root, precision_override=None):
    def run():
        import frdexp.training as training
        return _forced_precision_call(
            lambda: training.train_hybrid_model(
                key, hspec, enc, frame, text_col, features, feature_names,
                scaler, seed, config, root
            ),
            precision_override,
        )
    return run


def _dispatch_single(dispatch, state, results_dir, protocol, key, spec, frame, text_col,
                     seed, config, scope):
    root = Path(results_dir) / "models" / protocol / f"seed_{seed}" / key
    task = f"{protocol}:seed{seed}:single:{key}"
    print(f"\n=== {task} ===", flush=True)
    return dispatch(
        state, task,
        _single_training_callable(
            spec, frame, text_col, seed, config, root,
            precision_override=precision_override_for_family(spec.family),
        ),
        payload={"protocol": protocol, "model": key, "seed": seed,
                 "hf_id": spec.hf_id, "scope": scope,
                 "precision_policy": precision_override_for_family(spec.family) or "runtime_default"},
    )


def _dispatch_hybrid(dispatch, state, results_dir, protocol, key, hspec, enc, frame,
                     text_col, features, feature_names, scaler, seed, config, scope):
    root = Path(results_dir) / "models" / protocol / f"seed_{seed}" / key
    task = f"{protocol}:seed{seed}:hybrid:{key}"
    print(f"\n=== {task} ===", flush=True)
    return dispatch(
        state, task,
        _hybrid_training_callable(
            key, hspec, enc, frame, text_col, features, feature_names, scaler,
            seed, config, root, precision_override=precision_override_for_family(enc.family),
        ),
        payload={"protocol": protocol, "model": key, "seed": seed,
                 "encoder": enc.hf_id, "hf_id": enc.hf_id, "scope": scope,
                 "sentiment_features": "profile_9d",
                 "fusion": "gated" if hspec.get("gated") else "late_concat",
                 "precision_policy": precision_override_for_family(enc.family) or "runtime_default"},
    )


def _load_bundle(results_dir: Path, protocol: str, seed: int, key: str):
    from frdexp.orchestrator import load_prediction_bundle
    return load_prediction_bundle(Path(results_dir) / "models" / protocol / f"seed_{seed}" / key)


def _validation_metrics(results_dir: Path, protocol: str, seed: int, keys):
    out = {}
    for key in keys:
        bundle = _load_bundle(results_dir, protocol, seed, key)
        if bundle:
            out[key] = dict(bundle["metrics"]["validation_default"])
    return out


def _write_representative_selection(results_dir: Path, protocol: str, seed: int,
                                    out_root: Path, backbone_candidates=BACKBONE_CANDIDATES):
    metrics = _validation_metrics(
        results_dir, protocol, seed,
        [x for pair in backbone_candidates.values() for x in pair]
    )
    selected = select_backbone_representatives(metrics, backbone_candidates)
    missing = [b for b in backbone_candidates if b not in selected]
    if missing:
        raise RuntimeError(f"Missing complete candidates for backbone(s): {missing}")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for backbone, choices in backbone_candidates.items():
        for key in choices:
            m = metrics.get(key)
            if not m:
                continue
            rows.append({
                "backbone": backbone, "candidate": key,
                "selected": key == selected[backbone],
                "validation_accuracy": m.get("accuracy"),
                "validation_f1": m.get("f1"),
                "validation_roc_auc": m.get("roc_auc"),
            })
    with (out_root / "representative_selection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    payload = {
        "protocol": protocol,
        "selection_data": "validation_only",
        "selected_by_backbone": selected,
        "representatives": list(selected.values()),
    }
    (out_root / "representative_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _se_rank_auc(y_true, prob):
    import numpy as np
    y = np.asarray(y_true, dtype=int); p = np.asarray(prob, dtype=float)
    pos = y == 1; n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(p, kind="mergesort"); sorted_p = p[order]
    ranks = np.empty(len(p), dtype=float); i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    rs = float(ranks[pos].sum())
    return (rs - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _se_simple_metrics(y_true, prob, threshold=0.5):
    import numpy as np
    y = np.asarray(y_true, dtype=int); p = np.asarray(prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-15, precision + recall)
    return {"accuracy": (tp + tn) / max(1, len(y)), "f1": f1, "roc_auc": _se_rank_auc(y, p)}


def se_rank_equal_soft_triples(y_true, probs, top_n=3):
    """Validation-only triple ranking. There is intentionally no test argument."""
    import numpy as np
    rows = []
    for members in all_triples(probs):
        p = np.mean(np.column_stack([np.asarray(probs[m], dtype=float) for m in members]), axis=1)
        m = _se_simple_metrics(y_true, p, 0.5)
        rows.append({"members": tuple(members),
                     "validation_accuracy": float(m["accuracy"]),
                     "validation_f1": float(m["f1"]),
                     "validation_roc_auc": float(m["roc_auc"])})
    rows.sort(key=lambda r: (r["validation_accuracy"], r["validation_f1"],
                             r["validation_roc_auc"], tuple(r["members"])), reverse=True)
    return rows if top_n is None else rows[:max(0, int(top_n))]


def se_build_selected_triple_ensembles(results_dir, protocol, seed, representatives, out_root, top_n=3):
    """Search exactly the representative pool on validation; touch test only after selection."""
    from frdexp.orchestrator import load_prediction_bundle, _align_probabilities, _save_ensemble_method
    from frdexp.ensembles import weighted_probability, optimize_soft_weights

    model_base = Path(results_dir) / "models" / protocol / f"seed_{seed}"
    bundles = {k: load_prediction_bundle(model_base / k) for k in representatives}
    bundles = {k: v for k, v in bundles.items() if v is not None}
    if len(bundles) != 7:
        raise RuntimeError(f"Expected exactly 7 complete backbone representatives; got {sorted(bundles)}")
    val_base, val_probs = _align_probabilities(bundles, "validation")
    all_ranked = se_rank_equal_soft_triples(val_base.y_true.to_numpy(dtype=int), val_probs, top_n=None)
    if len(all_ranked) != 35:
        raise RuntimeError(f"Expected 35 representative triples, got {len(all_ranked)}")
    selected = all_ranked[:min(int(top_n), len(all_ranked))]
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    search_csv = out_root / "triple_search_validation.csv"
    with search_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["rank", "member_1", "member_2", "member_3",
                  "validation_accuracy", "validation_f1", "validation_roc_auc"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, row in enumerate(all_ranked, 1):
            w.writerow({"rank": i, "member_1": row["members"][0],
                        "member_2": row["members"][1], "member_3": row["members"][2],
                        "validation_accuracy": row["validation_accuracy"],
                        "validation_f1": row["validation_f1"],
                        "validation_roc_auc": row["validation_roc_auc"]})

    test_base, test_probs = _align_probabilities(bundles, "test")
    selection = []
    for rank, row in enumerate(selected, 1):
        members = tuple(row["members"]); group = f"triple_rank_{rank:02d}__" + "__".join(members)
        vp = {m: val_probs[m] for m in members}; tp = {m: test_probs[m] for m in members}
        equal = {m: 1 / 3 for m in members}
        _save_ensemble_method(out_root / group / "equal_soft", group, "equal_soft", seed,
                              val_base, test_base, weighted_probability(vp, equal),
                              weighted_probability(tp, equal),
                              {"members": members, "weights": equal, "protocol": protocol,
                               "selection_data": "validation_only", "triple_rank": rank})
        weights = optimize_soft_weights(val_base.y_true.to_numpy(dtype=int), vp, step=.05)
        _save_ensemble_method(out_root / group / "weighted_soft", group, "weighted_soft", seed,
                              val_base, test_base, weighted_probability(vp, weights),
                              weighted_probability(tp, weights),
                              {"members": members, "weights": weights, "protocol": protocol,
                               "selection_data": "validation_only", "triple_rank": rank,
                               "weight_search_step": .05})
        selection.append({"rank": rank, "members": members, "validation_search": row,
                          "equal_soft_dir": str((out_root / group / "equal_soft").resolve()),
                          "weighted_soft_dir": str((out_root / group / "weighted_soft").resolve())})
    manifest = {"runner_version": SENTIMENT_RUNNER_VERSION, "protocol": protocol, "seed": seed,
                "selection_data": "validation_only", "candidate_pool": sorted(bundles),
                "combination_size": 3, "total_combinations": len(all_ranked),
                "selected_top_n": len(selection), "selected": selection,
                "search_csv": str(search_csv.resolve())}
    (out_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {"artifact_path": str((out_root / "selection_manifest.json").resolve()),
            "payload": {"selected": selection, "total_combinations": len(all_ranked)}}


def run_zero_shot_model_safe(key, root, opspam, opspam_features, config, seed, out):
    from frdexp.orchestrator import run_zero_shot_model
    return run_zero_shot_model(key, root, opspam, opspam_features, config, seed, out)


def se_build_external_triple_ensembles(results_dir, seed, frd_ensemble_root, zero_shot_root, out_root):
    import numpy as np
    import pandas as pd
    from frdexp.metrics import compute_metrics
    from frdexp.prediction import save_predictions
    frd_ensemble_root = Path(frd_ensemble_root); zero_shot_root = Path(zero_shot_root); out_root = Path(out_root)
    manifest = json.loads((frd_ensemble_root / "selection_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for selected in manifest["selected"]:
        rank = int(selected["rank"]); members = tuple(selected["members"]); pred_frames = {}; base = None
        for m in members:
            d = pd.read_csv(zero_shot_root / m / "predictions.csv").sort_values("sample_id").reset_index(drop=True)
            if base is None:
                base = d
            elif not np.array_equal(base.sample_id.to_numpy(), d.sample_id.to_numpy()) or not np.array_equal(base.y_true.to_numpy(), d.y_true.to_numpy()):
                raise RuntimeError(f"Zero-shot prediction alignment mismatch for {m}")
            pred_frames[m] = d.y_prob_cg.to_numpy(float)
        for method_name, source_dir_key in (("equal_soft", "equal_soft_dir"), ("weighted_soft", "weighted_soft_dir")):
            source_dir = Path(selected[source_dir_key])
            meta = json.loads((source_dir / "ensemble_meta.json").read_text(encoding="utf-8"))
            weights = {k: float(v) for k, v in meta["weights"].items()}; wsum = sum(weights.values())
            prob = sum(pred_frames[k] * weights[k] for k in members) / wsum
            threshold = float(meta["selected_threshold"]); group = f"triple_rank_{rank:02d}__" + "__".join(members)
            target = out_root / group / method_name; target.mkdir(parents=True, exist_ok=True)
            payload = {"protocol": "zero_shot_frd_to_opspam", "model": group, "method": method_name,
                       "seed": seed, "members": members, "weights": weights,
                       "source_threshold": threshold, "selection_data": "FRD_validation_only",
                       "default": compute_metrics(base.y_true, prob, .5),
                       "frd_validation_calibrated": compute_metrics(base.y_true, prob, threshold)}
            (target / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            frame = pd.DataFrame({"source_index": base.sample_id, "label_num": base.y_true})
            save_predictions(target / "predictions.csv", frame, prob, threshold, group, seed, None)
            rows.append(payload)
    summary = out_root / "external_triple_ensemble_summary.json"; summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"artifact_path": str(summary.resolve()), "payload": {"created": len(rows)}}


def _metrics_threshold(bundle):
    return float(bundle["metrics"].get("selected_threshold", .5))


def run_extension_statistics(results_dir, seed, frd_ensemble_root, out_root, bootstrap_seed=314159):
    import numpy as np
    import pandas as pd
    from frdexp.statistics import bootstrap_ci, mcnemar_test, paired_bootstrap_difference, holm_adjust
    results_dir = Path(results_dir); out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    pairs = list(BACKBONE_CANDIDATES.items())
    rows = []; pvals = []; ablations = []
    for backbone, (plain, augmented) in pairs:
        a = _load_bundle(results_dir, "frd", seed, plain); b = _load_bundle(results_dir, "frd", seed, augmented)
        if not a or not b:
            continue
        da = a["test"].sort_values("sample_id").reset_index(drop=True)
        db = b["test"].sort_values("sample_id").reset_index(drop=True)
        if not np.array_equal(da.sample_id.to_numpy(), db.sample_id.to_numpy()):
            raise RuntimeError(f"Prediction alignment mismatch for {plain} vs {augmented}")
        ta, tb = _metrics_threshold(a), _metrics_threshold(b)
        mc = mcnemar_test(da.y_true, (da.y_prob_cg >= ta).astype(int), (db.y_prob_cg >= tb).astype(int))
        pvals.append(mc["p"]); row = {"comparison": "sentiment_ablation", "backbone": backbone,
                                      "model_a": plain, "model_b": augmented, **mc}
        for metric in ("accuracy", "f1", "roc_auc"):
            d = paired_bootstrap_difference(da.y_true, da.y_prob_cg, db.y_prob_cg, metric, ta, tb,
                                            n_boot=1000, seed=bootstrap_seed)
            row[f"{metric}_diff_aug_minus_plain"] = -d["difference"]
            row[f"{metric}_ci_low_aug_minus_plain"] = -d["high"]
            row[f"{metric}_ci_high_aug_minus_plain"] = -d["low"]
        rows.append(row)
        ma = a["metrics"]["test_calibrated"]; mb = b["metrics"]["test_calibrated"]
        ar = {"backbone": backbone, "plain": plain, "sentiment": augmented}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"):
            ar[f"{metric}_plain"] = ma[metric]; ar[f"{metric}_sentiment"] = mb[metric]
            ar[f"{metric}_delta"] = mb[metric] - ma[metric]
        ablations.append(ar)
    if rows:
        adj = holm_adjust(pvals)
        for row, hp in zip(rows, adj):
            row["holm_p"] = hp; row["significant_0_05"] = bool(hp < .05)
    pd.DataFrame(rows).to_csv(out_root / "statistical_tests.csv", index=False)
    pd.DataFrame(ablations).to_csv(out_root / "backbone_sentiment_ablation.csv", index=False)

    # CIs for the seven selected representatives and the top selected ensemble methods.
    rep = _write_representative_selection(results_dir, "frd", seed, out_root / "_selection_for_stats")
    ci_rows = []
    for key in rep["representatives"]:
        b = _load_bundle(results_dir, "frd", seed, key); t = b["test"]; th = _metrics_threshold(b)
        for metric in ("accuracy", "f1", "roc_auc"):
            ci = bootstrap_ci(t.y_true, t.y_prob_cg, metric, th, n_boot=1000, seed=bootstrap_seed)
            ci_rows.append({"model": key, "metric": metric, **ci})
    manifest = json.loads((Path(frd_ensemble_root) / "selection_manifest.json").read_text(encoding="utf-8"))
    for selected in manifest.get("selected", []):
        for method in ("equal_soft", "weighted_soft"):
            d = Path(selected[f"{method}_dir"])
            mp = json.loads((d / "metrics.json").read_text(encoding="utf-8")); t = pd.read_csv(d / "test_predictions.csv")
            key = "ensemble::" + d.parent.name + "::" + method; th = float(mp.get("selected_threshold", .5))
            for metric in ("accuracy", "f1", "roc_auc"):
                ci = bootstrap_ci(t.y_true, t.y_prob_cg, metric, th, n_boot=1000, seed=bootstrap_seed)
                ci_rows.append({"model": key, "metric": metric, **ci})
    pd.DataFrame(ci_rows).to_csv(out_root / "confidence_intervals.csv", index=False)
    manifest_out = {"comparisons": len(rows), "backbones": list(BACKBONE_CANDIDATES),
                    "selection_data": "validation_only", "bootstrap_n": 1000}
    (out_root / "statistics_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    return {"artifact_path": str((out_root / "statistical_tests.csv").resolve()), "payload": manifest_out}


def _summary_csv(results_dir, protocol, seed, keys, out_csv):
    rows = []
    for key in keys:
        b = _load_bundle(results_dir, protocol, seed, key)
        if not b:
            continue
        m = b["metrics"]; row = {"model": key, "selected_threshold": m.get("selected_threshold")}
        for section in ("validation_default", "test_default", "test_calibrated"):
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "tn", "fp", "fn", "tp"):
                if metric in m.get(section, {}): row[f"{section}_{metric}"] = m[section][metric]
        rows.append(row)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", newline="", encoding="utf-8-sig") as f:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def _se_zero_shot_summary(zero_root: Path, keys, out_csv: Path):
    rows = []
    for key in keys:
        p = Path(zero_root) / key / "metrics.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8")); row = {"model": key}
        for section in ("default", "frd_validation_calibrated"):
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"):
                if metric in d.get(section, {}): row[f"{section}_{metric}"] = d[section][metric]
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fields = sorted({k for r in rows for k in r}); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def _best_ensemble_from_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); candidates = []
    for selected in manifest.get("selected", []):
        for method in ("equal_soft", "weighted_soft"):
            d = Path(selected[f"{method}_dir"]); mp = d / "metrics.json"
            if not mp.exists(): continue
            m = json.loads(mp.read_text(encoding="utf-8")); v = m["validation_default"]
            candidates.append((v["accuracy"], v["f1"], v["roc_auc"], method, d, m))
    if not candidates: return None
    return max(candidates, key=lambda x: x[:4])


def compare_previous_ensemble(results_dir: Path, seed: int, new_root: Path, out_csv: Path):
    old_manifest = Path(results_dir) / "no_large_study" / f"seed_{seed}" / "ensembles" / "frd" / "selection_manifest.json"
    new_manifest = Path(new_root) / "selection_manifest.json"
    rows = []
    for label, path in (("previous_no_large", old_manifest), ("sentiment_representative", new_manifest)):
        best = _best_ensemble_from_manifest(path)
        if not best: continue
        _, _, _, method, d, m = best
        meta = json.loads((d / "ensemble_meta.json").read_text(encoding="utf-8"))
        rows.append({"study": label, "method": method, "members": ";".join(meta["members"]),
                     "validation_accuracy": m["validation_default"]["accuracy"],
                     "test_accuracy_calibrated": m["test_calibrated"]["accuracy"],
                     "test_f1_calibrated": m["test_calibrated"]["f1"],
                     "test_roc_auc": m["test_calibrated"]["roc_auc"]})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def _write_report(path: Path, frd_rows, opspam_rows, zero_rows, frd_rep, opspam_rep, frd_manifest, failures):
    def pct(v): return f"{100*float(v):.2f}%" if v not in (None, "") else ""
    lines = ["# Sentiment Extension Study", "",
             "Large models were excluded. Each non-large backbone was compared against one 9-D sentiment-profile variant.",
             "Backbone representatives and three-model ensembles were selected using validation predictions only.", "",
             "## FRD backbone representatives", "",
             "| Backbone | Selected representative |", "|---|---|"]
    for b, k in frd_rep["selected_by_backbone"].items(): lines.append(f"| {b} | {k} |")
    lines += ["", "## FRD models", "", "| Model | Accuracy | F1 | ROC-AUC |", "|---|---:|---:|---:|"]
    for r in sorted(frd_rows, key=lambda x: float(x.get("test_default_accuracy", 0)), reverse=True):
        lines.append(f"| {r['model']} | {pct(r.get('test_default_accuracy'))} | {pct(r.get('test_default_f1'))} | {r.get('test_default_roc_auc','')} |")
    lines += ["", "## OpSpam fine-tuned models", "", "| Model | Accuracy | F1 | ROC-AUC |", "|---|---:|---:|---:|"]
    for r in sorted(opspam_rows, key=lambda x: float(x.get("test_default_accuracy", 0)), reverse=True):
        lines.append(f"| {r['model']} | {pct(r.get('test_default_accuracy'))} | {pct(r.get('test_default_f1'))} | {r.get('test_default_roc_auc','')} |")
    lines += ["", "## FRD → OpSpam zero-shot", "", "| Model | Accuracy | F1 | ROC-AUC |", "|---|---:|---:|---:|"]
    for r in sorted(zero_rows, key=lambda x: float(x.get("default_accuracy", 0)), reverse=True):
        lines.append(f"| {r['model']} | {pct(r.get('default_accuracy'))} | {pct(r.get('default_f1'))} | {r.get('default_roc_auc','')} |")
    lines += ["", "## Selected FRD representative triples", ""]
    for s in frd_manifest.get("selected", []): lines.append(f"- Rank {s['rank']}: {', '.join(s['members'])}")
    lines += ["", "## OpSpam backbone representatives", ""]
    for b, k in opspam_rep["selected_by_backbone"].items(): lines.append(f"- {b}: {k}")
    lines += ["", "## Run status", "", f"- Failed tasks: {', '.join(failures) if failures else 'none'}"]
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines), encoding="utf-8")


def sentiment_main(argv=None):
    parser = argparse.ArgumentParser(description="Complete no-large tasks + sentiment-profile extension")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-retries", type=int, default=1, choices=(0, 1))
    parser.add_argument("--max-task-hours", type=float, default=0)
    args = parser.parse_args(argv)
    project = args.project.expanduser().resolve()

    # Critical: do this before loading scientific packages or the supervisor.
    ensure_safe_interpreter(project)
    supervisor = _load_supervisor(project)
    frd, frd_text, frd_split = _se_read_split(project, "frd")
    opspam, opspam_text, opspam_split = _se_read_split(project, "opspam")

    print(f"FRD sentiment extension runner {SENTIMENT_RUNNER_VERSION}", flush=True)
    print("PROJECT:", project, flush=True)
    print("PYTHON:", sys.executable, flush=True)
    print("SUPERVISOR:", supervisor.VERSION, flush=True)
    print("SCOPE: missing no-large tasks + 5 new 9-D sentiment variants + representative ensembles + statistics", flush=True)
    print("EXCLUDED LARGE MODELS:", ", ".join(sorted(SENTIMENT_LARGE_KEYS)), flush=True)
    print("FRD SPLIT:", frd_split, flush=True); print("OPSPAM SPLIT:", opspam_split, flush=True)

    with supervisor.ProjectLock(project / ".frd_supervisor" / "runner.lock"):
        runtime, _ = supervisor.original_runtime(project)
        session = project / ".frd_supervisor" / "sessions" / ("sentext_" + supervisor.stamp())
        session.mkdir(parents=True, exist_ok=True)
        results_dir = project / "results"; results_dir.mkdir(parents=True, exist_ok=True)
        supervisor.backup_database(results_dir / "state.db", session / "state_before.sqlite3")
        os.environ.update(supervisor.worker_env(project)); sys.path.insert(0, str(runtime))

        from config import CONFIG
        from frdexp.features import FEATURE_NAMES
        from frdexp.model_registry import MODEL_SPECS, HYBRID_SPECS
        from frdexp.state import StateDB

        # Guardrail against accidental large training.
        if any(v[0] in SENTIMENT_LARGE_KEYS for v in NEW_SENTIMENT_VARIANTS.values()):
            raise RuntimeError("Large model leaked into sentiment extension.")
        for plain, augmented in BACKBONE_CANDIDATES.values():
            if plain in SENTIMENT_LARGE_KEYS or augmented in SENTIMENT_LARGE_KEYS:
                raise RuntimeError("Large model leaked into representative candidates.")

        custom_hybrids = dict(HYBRID_SPECS)
        for key, (encoder_key, gated) in NEW_SENTIMENT_VARIANTS.items():
            custom_hybrids[key] = {
                "paper_name": MODEL_SPECS[encoder_key].paper_name + " + Sentiment Profile",
                "encoder": encoder_key, "features": "profile", "gated": bool(gated),
            }
        target_hybrids = ["deberta_v3_sentiment_gated"] + list(NEW_SENTIMENT_VARIANTS)
        all_profile_keys = sorted({x for pair in BACKBONE_CANDIDATES.values() for x in pair})

        frd_features, frd_scaler = _se_build_features(results_dir, "frd", frd, frd_text)
        opspam_features, opspam_scaler = _se_build_features(results_dir, "opspam", opspam, opspam_text)
        dispatch = supervisor.Dispatch(project, runtime, session,
                                       native_retries=args.native_retries,
                                       timeout=args.max_task_hours * 3600)
        state = StateDB(results_dir / "state.db")
        out_root = results_dir / "sentiment_extension" / f"seed_{args.seed}"
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            print("\n===== COMPLETE KNOWN MISSING TASKS =====", flush=True)
            _dispatch_hybrid(dispatch, state, results_dir, "frd", "deberta_v3_sentiment_gated",
                             custom_hybrids["deberta_v3_sentiment_gated"], MODEL_SPECS["deberta_v3_base"],
                             frd, frd_text, frd_features, FEATURE_NAMES, frd_scaler,
                             args.seed, CONFIG, "sentiment_extension")
            _dispatch_single(dispatch, state, results_dir, "opspam", "deberta", MODEL_SPECS["deberta"],
                             opspam, opspam_text, args.seed, CONFIG, "sentiment_extension")

            print("\n===== TRAIN FIVE NEW SENTIMENT VARIANTS =====", flush=True)
            for protocol, frame, text_col, features, scaler in (
                ("frd", frd, frd_text, frd_features, frd_scaler),
                ("opspam", opspam, opspam_text, opspam_features, opspam_scaler),
            ):
                print(f"\n--- {protocol.upper()} sentiment variants ---", flush=True)
                for key in NEW_SENTIMENT_VARIANTS:
                    hspec = custom_hybrids[key]; enc = MODEL_SPECS[hspec["encoder"]]
                    _dispatch_hybrid(dispatch, state, results_dir, protocol, key, hspec, enc,
                                     frame, text_col, features, FEATURE_NAMES, scaler,
                                     args.seed, CONFIG, "sentiment_extension")

            print("\n===== FRD -> OPSPAM ZERO-SHOT FOR ALL PLAIN/PROFILE CANDIDATES =====", flush=True)
            zero_root = results_dir / "external_validation" / f"seed_{args.seed}" / "zero_shot"
            for key in all_profile_keys:
                root = results_dir / "models" / "frd" / f"seed_{args.seed}" / key
                task = f"opspam_zero_shot:seed{args.seed}:{key}"
                dispatch(
                    state, task,
                    lambda key=key, root=root: run_zero_shot_model_safe(
                        key, root, opspam, opspam_features, CONFIG, args.seed, zero_root / key),
                    payload={"protocol": "zero_shot_frd_to_opspam", "model": key,
                             "seed": args.seed, "scope": "sentiment_extension"},
                )

            print("\n===== VALIDATION-ONLY BACKBONE REPRESENTATIVE SELECTION =====", flush=True)
            frd_rep = _write_representative_selection(results_dir, "frd", args.seed, out_root / "representatives" / "frd")
            opspam_rep = _write_representative_selection(results_dir, "opspam", args.seed, out_root / "representatives" / "opspam")
            print("FRD representatives:", frd_rep["selected_by_backbone"], flush=True)
            print("OpSpam representatives:", opspam_rep["selected_by_backbone"], flush=True)

            frd_ens_root = out_root / "ensembles" / "frd"
            opspam_ens_root = out_root / "ensembles" / "opspam"
            dispatch(
                state, f"sentext:seed{args.seed}:frd_representative_triples",
                lambda: se_build_selected_triple_ensembles(
                    results_dir, "frd", args.seed, frd_rep["representatives"], frd_ens_root, top_n=3),
                payload={"seed": args.seed, "selection_data": "validation_only",
                         "candidate_policy": "one_per_backbone", "expected_combinations": 35},
                skip_complete=False,
            )
            dispatch(
                state, f"sentext:seed{args.seed}:opspam_representative_triples",
                lambda: se_build_selected_triple_ensembles(
                    results_dir, "opspam", args.seed, opspam_rep["representatives"], opspam_ens_root, top_n=3),
                payload={"seed": args.seed, "selection_data": "validation_only",
                         "candidate_policy": "one_per_backbone", "expected_combinations": 35},
                skip_complete=False,
            )
            dispatch(
                state, f"sentext:seed{args.seed}:external_representative_triples",
                lambda: se_build_external_triple_ensembles(
                    results_dir, args.seed, frd_ens_root, zero_root, out_root / "external_triple_ensembles"),
                payload={"seed": args.seed, "selection_data": "FRD_validation_only"},
                skip_complete=False,
            )
            dispatch(
                state, f"sentext:seed{args.seed}:statistics",
                lambda: run_extension_statistics(
                    results_dir, args.seed, frd_ens_root, out_root / "statistics", CONFIG.bootstrap_seed),
                payload={"seed": args.seed, "scope": "sentiment_extension",
                         "comparisons": "7 plain-vs-sentiment backbones"},
                skip_complete=False,
            )
        finally:
            state.close()

        frd_rows = _summary_csv(results_dir, "frd", args.seed, all_profile_keys,
                                out_root / "frd_plain_vs_sentiment.csv")
        opspam_rows = _summary_csv(results_dir, "opspam", args.seed, all_profile_keys,
                                   out_root / "opspam_plain_vs_sentiment.csv")
        zero_rows = _se_zero_shot_summary(zero_root, all_profile_keys,
                                       out_root / "opspam_zero_shot_plain_vs_sentiment.csv")
        # Reload selections after worker tasks.  If an upstream task failed, still
        # emit an INCOMPLETE run summary instead of losing the diagnostic record.
        frd_rep = read_json_if_exists(
            out_root / "representatives" / "frd" / "representative_selection.json",
            {"selected_by_backbone": {}, "representatives": []},
        )
        opspam_rep = read_json_if_exists(
            out_root / "representatives" / "opspam" / "representative_selection.json",
            {"selected_by_backbone": {}, "representatives": []},
        )
        frd_manifest = read_json_if_exists(
            out_root / "ensembles" / "frd" / "selection_manifest.json",
            {"selected": [], "total_combinations": 0},
        )
        compare_previous_ensemble(results_dir, args.seed, out_root / "ensembles" / "frd",
                                  out_root / "ensemble_comparison_previous_vs_sentiment.csv")
        failed = sorted(set(dispatch.failures))
        report = out_root / "paper_ready_summary.md"
        _write_report(report, frd_rows, opspam_rows, zero_rows, frd_rep, opspam_rep, frd_manifest, failed)
        summary = {
            "runner_version": SENTIMENT_RUNNER_VERSION,
            "status": "COMPLETE" if not failed else "INCOMPLETE",
            "safe_python": str(Path(sys.executable).resolve()),
            "large_models_excluded": sorted(SENTIMENT_LARGE_KEYS),
            "new_sentiment_variants": NEW_SENTIMENT_VARIANTS,
            "backbone_candidates": BACKBONE_CANDIDATES,
            "frd_representatives": frd_rep["selected_by_backbone"],
            "opspam_representatives": opspam_rep["selected_by_backbone"],
            "frd_triple_combinations": frd_manifest.get("total_combinations"),
            "failed_tasks": failed,
            "output_root": str(out_root.resolve()),
            "paper_ready_summary": str(report.resolve()),
            "session": str(session.resolve()),
        }
        supervisor.atomic_json(out_root / "run_summary.json", summary)
        supervisor.atomic_json(session / "sentiment_extension_summary.json", summary)
        print("\n===== SENTIMENT EXTENSION SUMMARY =====", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
        print("\nRESULTS:", out_root, flush=True)
        return 0 if not failed else 2

# =============================================================================
# 4. Environment setup (formerly FRD_Create_Env.py)
# =============================================================================
# -*- coding: utf-8 -*-

CORE = '''torch==2.5.1+cu118
transformers==4.48.3
accelerate==1.2.1
peft==0.14.0
sentencepiece==0.2.0
protobuf==5.29.3
safetensors==0.5.3
huggingface-hub==0.28.1
tokenizers==0.21.0
numpy==1.26.4
pandas==2.2.3
scipy==1.11.4
scikit-learn==1.4.2
textblob==0.18.0.post0
matplotlib==3.9.4
psutil==6.1.1
joblib==1.4.2
fsspec==2024.5.0
sympy==1.13.1
networkx==3.2.1
'''
CUDA_INDEX = 'https://download.pytorch.org/whl/cu118'
PYPI_INDEX = 'https://pypi.org/simple'


def env_run(command, **kwargs):
    print('[CMD]',subprocess.list2cmdline([str(x) for x in command]),flush=True)
    return subprocess.run([str(x) for x in command],check=True,**kwargs)


def installation_commands(python, constraints, requirements):
    # The explicit local CUDA version remains a constraint for the second install.
    return [
        [str(python),'-m','pip','install','pip==25.3'],
        [str(python),'-m','pip','install','--only-binary=:all:',
         '--index-url',CUDA_INDEX,'-c',str(constraints),'torch==2.5.1+cu118'],
        [str(python),'-m','pip','install','--only-binary=:all:',
         '--index-url',PYPI_INDEX,'-c',str(constraints),'-r',str(requirements)],
        [str(python),'-m','pip','check'],
    ]


def env_main(argv=None):
    parser=argparse.ArgumentParser(description='Create the pinned Transformer environment.')
    parser.add_argument('--run',action='store_true',help='Run the whole Transformer study after environment verification.')
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(argv)
    project=args.project.resolve()
    runner=SELF
    target=project/'.venv_frd_safe'
    marker=target/'.created_for_frd_supervisor'
    if target.exists() and not marker.is_file():
        raise RuntimeError('Existing .venv_frd_safe has no ownership marker; refusing to alter it.')
    base=[sys.executable]
    if os.name=='nt' and shutil.which('py'):
        probe=subprocess.run(['py','-3.11','-c','import sys;print(sys.executable)'],
                             capture_output=True,text=True)
        if probe.returncode==0:
            base=[probe.stdout.strip()]
    probe=subprocess.run(base+['-c','import sys;print("%d.%d"%sys.version_info[:2])'],
                         capture_output=True,text=True,check=True)
    version=tuple(int(x) for x in probe.stdout.strip().split('.'))
    if not (3,9)<=version<(3,13):
        raise RuntimeError('This version set needs Python 3.9-3.12; preferably an installed Python 3.11.')
    print('NEW ENVIRONMENT:',target,flush=True)
    print('OLD .venv AND ALL MODEL RESULTS WILL NOT BE DELETED OR UPGRADED.',flush=True)
    if not target.exists():
        target.mkdir()
        marker.write_text('FRD isolated environment\n',encoding='utf8')
    python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():
        env_run(base+['-m','venv',str(target)])
    constraints=target/'core_constraints.txt'
    requirements=target/'requirements_core.txt'
    constraints.write_text(CORE,encoding='utf8')
    requirements.write_text('\n'.join(x for x in CORE.splitlines() if not x.startswith('torch=='))+'\n',encoding='utf8')
    ready=target/'frd_environment_ready.json'
    signature=hashlib.sha256(CORE.encode()).hexdigest()
    previous=json.loads(ready.read_text()) if ready.is_file() else {}
    if previous.get('core_sha256')!=signature:
        for command in installation_commands(python,constraints,requirements): env_run(command)
    env=os.environ.copy()
    env.update(USE_TORCH='1',USE_TF='0',USE_FLAX='0',PYTHONIOENCODING='utf-8')
    smoke = (
        'import torch,numpy,pandas,sklearn,transformers,accelerate,peft; '
        'print("Torch:",torch.__version__); print("CUDA:",torch.version.cuda); '
        'assert torch.__version__=="2.5.1+cu118", "Wrong PyTorch build"; '
        'assert torch.cuda.is_available(), "CUDA unavailable; installation is not marked ready"; '
        'print("GPU:",torch.cuda.get_device_name(0)); '
        'print("CUDA kernel:",torch.ones(2,device="cuda").sum().item()); '
        'print("Basic CUDA/import smoke test passed; this is NOT a full model-training test.")'
    )
    env_run([python,'-u','-c',smoke],env=env)
    env_run([python,'-m','pip','check'])
    freeze=subprocess.run([str(python),'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (target/'installed_freeze.txt').write_text(freeze.stdout,encoding='utf8')
    ready.write_text(json.dumps({'core_sha256':signature,'time':datetime.now(timezone.utc).isoformat(),
                                'python':str(python),'base_python':base[0]},indent=2),encoding='utf8')
    command=[str(python),'-u',str(runner),'all','--project',str(project)]
    if args.run:
        return subprocess.run(command,cwd=str(project),env=env).returncode
    print('Next:',subprocess.list2cmdline(command),flush=True)
    return 0

# =============================================================================
# 5. Plain DeBERTa-v3 (formerly run_deberta_v3_only.py)
# =============================================================================
def train_deberta_v3_plain(project):
    """Plain DeBERTa-v3 in FP32 with its SentencePiece tokenizer.

    This is the original run_deberta_v3_only.py, unchanged except for the
    project/runtime paths. Its validation and test predictions are then
    regenerated from the saved weights by the supervisor (reconciliation)
    when the train stage reaches this model.
    """
    import json
    import math
    import os
    import time
    from pathlib import Path

    # Must be set before importing torch/CUDA.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset

    PROJECT = Path(project).resolve()
    RUNTIME = original_runtime(PROJECT)[0]
    import sys
    sys.path.insert(0, str(RUNTIME))

    from config import CONFIG
    from frdexp.model_registry import MODEL_SPECS
    from frdexp.resources import build_training_profiles


    class ReviewDataset(Dataset):
        def __init__(self, frame, tokenizer, text_col, max_length):
            self.frame = frame.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.text_col = text_col
            self.max_length = max_length

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            row = self.frame.iloc[i]
            x = self.tokenizer(
                str(row[self.text_col]),
                truncation=True,
                max_length=self.max_length,
            )
            x["labels"] = int(row["label_num"])
            return x


    def metrics_from_logits(logits, labels):
        logits = np.asarray(logits)
        labels = np.asarray(labels).astype(int)
        pred = np.argmax(logits, axis=-1)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        n = max(1, len(labels))
        acc = (tp + tn) / n
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        tpr = recall
        tnr = tn / max(1, tn + fp)
        balanced_acc = (tpr + tnr) / 2
        denom = math.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = ((tp * tn) - (fp * fn)) / denom
        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_acc),
            "mcc": float(mcc),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }


    def main():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available.")

        print("CLEAN DeBERTa-v3 launcher", flush=True)
        print("Torch:", torch.__version__, flush=True)
        print("CUDA:", torch.version.cuda, flush=True)
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

        split_file = PROJECT / "results" / "splits" / "frd_split.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Existing FRD split not found: {split_file}")

        frame = pd.read_csv(split_file)
        required = {"text_", "label_num", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"FRD split is missing columns: {sorted(missing)}")

        train = frame[frame["split"] == "train"].copy()
        val = frame[frame["split"] == "validation"].copy()
        test = frame[frame["split"] == "test"].copy()
        print(f"Splits: train={len(train)}, validation={len(val)}, test={len(test)}", flush=True)

        spec = MODEL_SPECS["deberta_v3_base"]
        seed = CONFIG.train_seeds[0]
        root = PROJECT / "results" / "models" / "frd" / f"seed_{seed}" / "deberta_v3_base"
        root.mkdir(parents=True, exist_ok=True)
        attempt = root / "full_primary"
        checkpoint_dir = attempt / "checkpoints"
        artifact = attempt / "artifact"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Exact RTX-3060 profile from the project resource logic, without importing training.py.
        profiles = build_training_profiles(
            float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3),
            spec.large,
            CONFIG.effective_batch_size,
        )
        profile = profiles[0]
        print(
            f"Profile: {profile.name}; micro_batch={profile.micro_batch}; "
            f"grad_accum={profile.grad_accum}; adaptation={profile.adaptation}",
            flush=True,
        )

        # Deliberately DO NOT import frdexp.data, frdexp.training, or frdexp.metrics.
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        from transformers.trainer_utils import get_last_checkpoint

        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            cache_dir=str(CONFIG.model_cache_dir) if CONFIG.model_cache_dir else None,
            use_fast=False,
        )
        print("Tokenizer OK", flush=True)

        print("Loading DeBERTa-v3-base...", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_id,
            num_labels=2,
            cache_dir=str(CONFIG.model_cache_dir) if CONFIG.model_cache_dir else None,
        )
        print("Model CPU load OK", flush=True)
        model.to("cuda")
        print("Model GPU load OK", flush=True)

        train_ds = ReviewDataset(train, tokenizer, "text_", CONFIG.max_length)
        val_ds = ReviewDataset(val, tokenizer, "text_", CONFIG.max_length)
        test_ds = ReviewDataset(test, tokenizer, "text_", CONFIG.max_length)

        steps_per_epoch = max(
            1, math.ceil(len(train) / (profile.micro_batch * profile.grad_accum))
        )
        eval_steps = max(25, steps_per_epoch // max(1, CONFIG.evals_per_epoch))

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            return metrics_from_logits(logits, labels)

        args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            overwrite_output_dir=False,
            num_train_epochs=CONFIG.max_epochs,
            per_device_train_batch_size=profile.micro_batch,
            per_device_eval_batch_size=max(1, min(32, profile.micro_batch * 4)),
            gradient_accumulation_steps=profile.grad_accum,
            learning_rate=spec.learning_rate,
            weight_decay=CONFIG.weight_decay,
            warmup_ratio=CONFIG.warmup_ratio,
            lr_scheduler_type="linear",
            logging_steps=min(CONFIG.logging_steps, eval_steps),
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=CONFIG.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=seed,
            data_seed=seed,
            max_grad_norm=CONFIG.gradient_clip,
            dataloader_num_workers=CONFIG.num_workers,
            report_to=[],
            fp16=False,
            bf16=False,
            remove_unused_columns=True,
            save_safetensors=True,
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=CONFIG.early_stopping_patience
                )
            ],
        )

        last = get_last_checkpoint(str(checkpoint_dir))
        print("Starting training...", flush=True)
        if last:
            print("Resuming:", last, flush=True)

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.train(resume_from_checkpoint=last if last else None)
        train_seconds = time.perf_counter() - start
        print("TRAINING FINISHED", flush=True)

        print("Running validation/test prediction...", flush=True)
        val_out = trainer.predict(val_ds)
        test_out = trainer.predict(test_ds)

        val_metrics = metrics_from_logits(val_out.predictions, val_out.label_ids)
        test_metrics = metrics_from_logits(test_out.predictions, test_out.label_ids)

        artifact.mkdir(parents=True, exist_ok=True)
        trainer.save_model(artifact)
        tokenizer.save_pretrained(artifact / "tokenizer")

        resource = {
            "profile": profile.name,
            "micro_batch": profile.micro_batch,
            "gradient_accumulation": profile.grad_accum,
            "effective_batch": profile.micro_batch * profile.grad_accum,
            "adaptation": profile.adaptation,
            "precision": "fp32",
            "learning_rate": spec.learning_rate,
            "weight_decay": CONFIG.weight_decay,
            "warmup_ratio": CONFIG.warmup_ratio,
            "max_length": CONFIG.max_length,
            "max_epochs": CONFIG.max_epochs,
            "train_seconds": train_seconds,
            "global_step": int(getattr(trainer.state, "global_step", 0)),
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "best_metric": getattr(trainer.state, "best_metric", None),
            "final_epoch": getattr(trainer.state, "epoch", None),
            "peak_vram_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "peak_reserved_vram_gb": float(torch.cuda.max_memory_reserved() / (1024 ** 3)),
        }

        payload = {
            "model": spec.key,
            "seed": seed,
            "validation_default": val_metrics,
            "selected_threshold": 0.5,
            "test_default": test_metrics,
            "test_calibrated": test_metrics,
            "resource": resource,
            "artifact_path": str(artifact),
            "profile": profile.name,
            "result_dir": str(attempt),
        }

        (attempt / "metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        (artifact / "artifact_meta.json").write_text(
            json.dumps(
                {
                    "type": "classifier",
                    "model_key": spec.key,
                    "paper_name": spec.paper_name,
                    "hf_id": spec.hf_id,
                    "adaptation": profile.adaptation,
                    "seed": seed,
                    "resource": resource,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "selected_result.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        print("=== DEBERTA-V3-BASE COMPLETED ===", flush=True)
        print(json.dumps(payload, indent=2), flush=True)
    return main()


# =============================================================================
# 6. Command line
# =============================================================================
def make_split(project):
    """Extract the runtime and create the persisted FRD and OpSpam splits (no training)."""
    runtime, _ = original_runtime(project)
    os.environ.update(worker_env(project))
    sys.path.insert(0, str(runtime))
    import run_all
    return int(run_all.main(['--dry-run']) or 0)


def print_status(project):
    report = audit(project)
    counts = {}
    for row in report['checks']:
        counts[row['audit_status']] = counts.get(row['audit_status'], 0) + 1
        print(row['task_key'], '|', row['audit_status'])
    print('SUMMARY:', counts or 'no recorded tasks')
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--worker']:
        return task_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description='FRD Transformer study (see module docstring).')
    parser.add_argument('command', choices=['setup', 'split', 'deberta-v3', 'train', 'all', 'status'])
    parser.add_argument('--project', type=Path, default=SELF.parent)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--native-retries', type=int, default=1, choices=(0, 1))
    parser.add_argument('--max-task-hours', type=float, default=0)
    args = parser.parse_args(argv)
    project = args.project.expanduser().resolve()

    if args.command == 'setup':
        return env_main(['--project', str(project)])
    if args.command == 'status':
        return print_status(project)
    ensure_safe_interpreter(project, argv)
    stage_args = ['--project', str(project), '--seed', str(args.seed),
                  '--native-retries', str(args.native_retries), '--max-task-hours', str(args.max_task_hours)]
    codes = []
    if args.command in ('split', 'all'):
        print('\n===== SPLIT =====', flush=True)
        codes.append(make_split(project))
        if codes[-1]:
            return codes[-1]
    if args.command in ('deberta-v3', 'all'):
        print('\n===== PLAIN DEBERTA-V3 =====', flush=True)
        done = project / 'results' / 'models' / 'frd' / f'seed_{args.seed}' / 'deberta_v3_base' / 'selected_result.json'
        if done.is_file():
            print('Already trained:', done, flush=True)
        else:
            codes.append(int(train_deberta_v3_plain(project) or 0))
    if args.command in ('train', 'all'):
        print('\n===== NO-LARGE STUDY =====', flush=True)
        codes.append(int(nolarge_main(stage_args) or 0))
        print('\n===== SENTIMENT EXTENSION =====', flush=True)
        codes.append(int(sentiment_main(stage_args) or 0))
    return max(codes) if codes else 0



# =============================================================================
# 7. Embedded runtime (formerly inside FRD_Revision_AllInOne.py): zip archive of
#    config.py, run_all.py and frdexp/*.py, checked against PAYLOAD_SHA256
# =============================================================================
PAYLOAD_SHA256 = "8ae961477fbd4991518d297cd3deff5c65b0d72bb4285b129022ec953323b2e7"
PAYLOAD_B64 = """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"""


if __name__ == '__main__':
    raise SystemExit(main())
