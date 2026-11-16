#!/usr/bin/env python3
"""Durable per-user jobs. Task data stays in local private state, never DT."""
import argparse, json, os, pathlib, re, shlex, shutil, subprocess, sys, time
os.umask(0o077)
ROOT = pathlib.Path(__file__).resolve().parent / 'task_state'
TMUX = shutil.which('tmux') or next((p for p in ['/opt/homebrew/bin/tmux', '/usr/bin/tmux', str(pathlib.Path.home()/'.local/bin/tmux')] if os.path.isfile(p)), None)
def mux(*args, **kw):
    if not TMUX: raise SystemExit('tmux missing: refusing unprotected execution')
    return subprocess.run([TMUX, '-f', '/dev/null', '-S', 'tasks.sock', *args], cwd=ROOT.parent, **kw)
def alive(name):
    return mux('has-session', '-t', '='+name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
def status(d):
    if (d/'exit.json').exists(): return json.loads((d/'exit.json').read_text())
    return {'state': 'running' if alive(d.name) else 'interrupted', 'name': d.name}
def worker(d):
    cfg=json.loads((d/'job.json').read_text())
    with (d/'output.log').open('ab', buffering=0) as log:
        try: rc=subprocess.call(cfg['argv'], cwd=cfg['cwd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        except Exception as e:
            log.write((str(e)+'\n').encode()); rc=127
    tmp=d/'exit.tmp'
    tmp.write_text(json.dumps({'state':'succeeded' if rc==0 else 'failed','exit_code':rc,'finished':time.time()}))
    tmp.replace(d/'exit.json')
if len(sys.argv)>1 and sys.argv[1]=='_worker':
    worker(pathlib.Path(sys.argv[2])); sys.exit()
p=argparse.ArgumentParser(description='Durable jobs: start NAME --cwd DIR -- COMMAND ARGS; status/log/attach NAME; list')
p.add_argument('action', choices=['start','status','log','attach','list'])
p.add_argument('name', nargs='?')
p.add_argument('--cwd', default=os.getcwd())
# Command is deliberately separated from options; shell expansion is caller-controlled.
args=sys.argv[1:]; command=[]
if '--' in args:
    idx=args.index('--'); command=args[idx+1:]; args=args[:idx]
a=p.parse_args(args)
ROOT.mkdir(parents=True, exist_ok=True)
if a.action=='list':
    for d in sorted(ROOT.iterdir()):
        if d.is_dir(): print(d.name, json.dumps(status(d)))
    sys.exit()
if not a.name or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',a.name): p.error('NAME must be 1–64 letters, digits, underscores or hyphens')
d=ROOT/a.name
if a.action=='start':
    if not TMUX: p.error('tmux missing: refusing unprotected execution')
    if not command: p.error('provide -- COMMAND ARGS')
    cwd=pathlib.Path(a.cwd).expanduser().resolve()
    if not cwd.is_dir(): p.error('cwd is not a directory')
    try: d.mkdir()
    except FileExistsError: p.error('name already exists; inspect status/log, or choose a new name (no duplicate launched)')
    (d/'job.json').write_text(json.dumps({'argv':command,'cwd':str(cwd),'created':time.time()}))
    runner=shlex.join([sys.executable,str(pathlib.Path(__file__).resolve()),'_worker',str(d)])
    result=mux('new-session','-d','-s',a.name,runner, capture_output=True,text=True)
    if result.returncode:
        (d/'exit.json').write_text(json.dumps({'state':'launch_failed','error':result.stderr}))
        raise SystemExit(result.stderr)
    print(json.dumps({'name':a.name,'state':'submitted','directory':str(d)}))
else:
    if not d.is_dir(): p.error('unknown task')
    if a.action=='status': print(json.dumps(status(d)))
    elif a.action=='log':
        f=d/'output.log'
        if f.exists(): subprocess.run(['tail','-n','80',str(f)])
    elif a.action=='attach':
        if not alive(a.name): raise SystemExit('Task is not running; use status/log')
        os.execv(TMUX,[TMUX,'-S',str(ROOT.parent / 'tasks.sock'),'attach-session','-t','='+a.name])
