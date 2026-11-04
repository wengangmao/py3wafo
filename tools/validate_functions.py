"""Run all tests and generate passed/failed/untested/unimplemented reports."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
temporary = root/'.build-tmp'
temporary.mkdir(exist_ok=True)
environment = dict(os.environ,TEMP=str(temporary),TMP=str(temporary))
environment['MPLCONFIGDIR'] = str(temporary/'matplotlib')
paths = [str(root/'src'),str(root/'tools')]
# Compiled dependencies are interpreter specific; prefer a matching folder.
dependencies = root/f'.test-deps-py{sys.version_info[0]}{sys.version_info[1]}'
if not dependencies.is_dir():
    dependencies = root/'.test-deps'
if dependencies.is_dir():
    paths.append(str(dependencies))
if environment.get('PYTHONPATH'):
    paths.append(environment['PYTHONPATH'])
environment['PYTHONPATH'] = os.pathsep.join(paths)
result = subprocess.run([sys.executable,'-m','pytest','-p','function_validation_plugin',
    '-q','--junitxml=docs/migration/test_results.xml',*sys.argv[1:]],cwd=root,env=environment)
raise SystemExit(result.returncode)
