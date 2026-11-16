import os,sys,pathlib,resource,signal
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'shared/src'))
resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));signal.alarm(1800)
import runner
runner.run('E2')
