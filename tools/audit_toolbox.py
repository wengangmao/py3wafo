"""Rebuild exhaustive source inventories and conversion coverage from local files.

Run from any directory: python tools/audit_toolbox.py [path/to/wafo_matlab]
No source binaries are executed. Dependency matches are static evidence only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import importlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from wafo._native_catalog import replacement_for

FAMILIES = {
    'down2cc': 'Downcrossing intensity to cycle-count/damage conversion; source down2cc.f is absent',
    'findcross': 'Level crossing indices; C MEX gateway',
    'findrfc': 'Rainflow filtering of turning points; C MEX gateway',
    'csort': 'Stable counting sort of integer data; C MEX gateway',
    'disufq': 'Second-order sum- and difference-frequency wave interactions; C MEX gateway',
    'rind': 'Gaussian expectations/integrals with bounds and derivative/Jacobian weights',
    'alanpab': 'Alternative Gaussian integration engine used by RIND MEX gateways',
    'jacobmod': 'Jacobian/derivative weight support for Gaussian wave integrals',
    'intmodule': 'Numerical integration support module for RIND and wave-density drivers',
    'mvn': 'Multivariate normal rectangle probability integration',
    'mvt': 'Multivariate Student-t/normal probability integration',
    'mvdist': 'Multivariate distribution integration and Fortran module support',
    'sp2acdf': 'Spectrum to crest-amplitude distribution driver',
    'sp2mm': 'Spectrum to joint minimum/maximum or minimum/maximum/period density driver',
    'sp2tccpdf': 'Spectrum to crest-to-crest period and crest characteristic density driver',
    'sp2thpdf': 'Spectrum to joint period/height density driver',
    'minmax': 'Gaussian process minimum/maximum distribution executable driver',
    'twog': 'Bivariate Gaussian density/integration support for minmax',
    'simduff': 'Duffing oscillator numerical simulation driver',
    'd1mach': 'Floating point machine constants for numerical integration',
    'dsvdc': 'Singular-value decomposition numerical support',
}


def inferred_users(stem, direct):
    """Source/build-family associations, kept separate from literal call evidence."""
    if any(s in stem for s in ('rind','alanpab','intmodule','jacobmod')):
        return ['trgauss/rind.m','trgauss/spec2Acdf.m','trgauss/spec2AcAt.m',
                'trgauss/spec2mmtpdf.m','trgauss/spec2tccpdf.m','trgauss/spec2thpdf.m','trgauss/spec2tpdf.m','trgauss/spec2tpdf2.m']
    if 'sp2acdf' in stem:
        return ['trgauss/spec2Acdf.m','trgauss/spec2tpdf.m']
    if 'sp2mm' in stem:
        return ['trgauss/spec2mmtpdf.m']
    if 'sp2tccpdf' in stem:
        return ['trgauss/spec2tccpdf.m','trgauss/spec2AcAt.m']
    if 'sp2thpdf' in stem:
        return ['trgauss/spec2thpdf.m']
    if stem in ('minmax','twog'):
        return ['trgauss/private/wminmax.m']
    if 'mvdist' in stem or 'mvt' in stem:
        return ['trgauss/mvnortpcprb.m (distribution-integration family; exact MEX branch needs review)']
    return direct


def description(path, text):
    low = path.stem.lower()
    for token, purpose in FAMILIES.items():
        if token in low:
            return purpose
    if path.suffix.lower() in ('.c', '.f', '.f90'):
        lines = [re.sub(r'^[cC*!/\s]+', '', s).strip() for s in text.splitlines()[:70]]
        return 'Native numerical source: '+next((s for s in lines if len(s)>25 and 'include' not in s.lower()), path.name)
    return ''


def classify(p, text):
    ext = p.suffix.lower()
    top = p.parts[0].lower()
    if ext in ('.c', '.f', '.f90'):
        return 'native-source', description(p, text)
    if ext in ('.dll', '.exe') or ext.startswith('.mex') or (top == 'exec' and not ext):
        return 'native-binary', description(p, text) or 'Platform executable/library; purpose requires matching driver/source'
    if ext in ('.html', '.chm', '.hhp', '.hhc', '.css', '.map', '.dot'):
        return 'documentation', 'Generated help/site navigation or documentation dependency graph'
    if ext in ('.png', '.gif', '.eps', '.ps', '.pstex', '.fig', '.pstex_t'):
        return 'figure', 'Image, plot, MATLAB figure, or publication drawing; data asset, not executable algorithm'
    if ext in ('.tex', '.sty', '.bib', '.bbl', '.blg', '.dvi', '.pdf'):
        return 'publication', 'Manual/paper/typesetting source or compiled publication'
    if ext in ('.mat', '.dat', '.asc', '.in', '.out', '.ncx') or p.name.startswith(('mon.', 'tue.')):
        return 'data', 'Stored measurements/model parameters, integration input/output, or binary scientific data/index'
    if ext in ('.scc', '.asv', '.mold', '.old', '.bak', '.m#', '.tpl~'):
        return 'legacy-metadata', 'Version-control metadata or backup/editor artifact'
    if ext == '.csh':
        return 'publication-script', 'C-shell helper for Xfig/LaTeX/EPS or PostScript/PDF publication conversion'
    if ext == '.txt':
        return 'text', next((s.strip() for s in text.splitlines() if s.strip()), 'Text documentation or numerical output')[:180]
    return 'unclassified-asset', 'Unknown/legacy asset; inspect before assuming it is executable code'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='?', type=Path, default=ROOT.parent/'wafo_matlab')
    args = parser.parse_args()
    source = args.source.resolve()
    docs = ROOT/'docs'/'migration'
    docs.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source.rglob('*') if p.is_file())
    matlab = {p.relative_to(source).as_posix(): p.read_text(encoding='latin1') for p in files if p.suffix.lower()=='.m'}
    # Ignore full-line comments for direct caller detection; string-based executable names remain.
    active = {k: '\n'.join(s.split('%', 1)[0] for s in v.splitlines() if not s.lstrip().startswith('%')).lower() for k,v in matlab.items()}
    callers = {}
    for k, v in active.items():
        for token in set(re.findall(r'[a-z_][a-z_0-9]*', v)):
            callers.setdefault(token, set()).add(k)
    records = []
    for p in files:
        if p.suffix.lower() == '.m':
            continue
        rel = p.relative_to(source)
        text = p.read_bytes()[:40000].decode('latin1') if p.suffix.lower() in ('.c','.f','.f90','.txt','.csh') else ''
        category, purpose = classify(rel, text)
        stem = p.stem.lower()
        stems = {stem}
        if stem.startswith('mex'):
            stems.add(stem[3:])
        # Variants such as sp2mM70.exe are literal names in MATLAB wrappers.
        direct = sorted(set().union(*(callers.get(s, set()) for s in stems)))
        native = category in ('native-source','native-binary')
        fallback = next((s for s in ('findcross','findrfc','csort','disufq','simduff') if s in stems), None)
        status = ('matlab_fallbacks/'+fallback+'.m') if fallback and native else (
            'Requires dedicated numerical port and parity validation; not mathematically impossible' if native else 'Asset/metadata: no MATLAB algorithm rewrite applicable')
        records.append(dict(path=rel.as_posix(), category=category, bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), functionality=purpose,
            direct_matlab_callers='; '.join(direct),
            associated_wafo_functions='; '.join(inferred_users(stem, direct)) if native else '',
            matlab_replacement=status,
            **(replacement_for(rel.as_posix()) if native else dict(
                python_api=('wafo.docs.convert_fig' if p.name=='ConvertFig.csh' else
                            'wafo.docs.convert_ps' if p.name=='ConvertToPDF.csh' else ''),
                python_status='python-orchestration' if p.suffix.lower()=='.csh' else 'asset-or-metadata',
                python_limitations=('Requires external format renderer fig2dev/Ghostscript; original hard-coded publishing moves removed'
                                    if p.suffix.lower()=='.csh' else 'No numerical algorithm to translate; original asset retained')))))
    with (docs/'non_matlab_files.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (docs/'non_matlab_files.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    native = [r for r in records if r['category'].startswith('native-')]
    (ROOT/'src/wafo/native_manifest.json').write_text(json.dumps(native,indent=2),encoding='utf-8')
    with (docs/'NATIVE_DEPENDENCIES.md').open('w', encoding='utf-8') as f:
        f.write('# Native file audit\n\nEvery non-MATLAB file is listed in `non_matlab_files.csv` and `.json`, with size, SHA-256, functionality, category and direct MATLAB callers. No binaries were run.\n\n')
        f.write('## Interpretation\n\nPython capability replacements are now associated with every native source/binary below. `ported` identifies direct numerical kernels; `algorithm-replacement` consolidates old integration engines into modern implementations; `capability-replacement-limited` identifies explicitly restricted Gaussian wave drivers; `reconstructed` identifies the source-missing down2cc algorithm. These statuses do not certify binary equivalence or full legacy option coverage. See `NATIVE_PYTHON_PORTS.md` and each Python docstring. The earlier pure MATLAB fallbacks remain in `extras/matlab_fallbacks`. Data, documents and images remain assets.\n\n')
        f.write('Callers below are conservative static name matches in active MATLAB lines. They exclude full-line comments, but do not resolve every dynamic executable name or Fortran link dependency. See `source/compilation.txt` for driver/link recipes. An empty entry means no direct name match, not that the file is unused. RIND, Jacobian and integration support are transitively needed by the spectrum-to-wave distribution drivers.\n\n')
        f.write('| File | Functionality | Direct MATLAB callers | Associated WAFO functions (family/link inference) | Python capability | Status | Limits |\n|---|---|---|---|---|---|---|\n')
        for r in native:
            f.write('| '+' | '.join(str(r[k]).replace('|','/') or 'No static match; see link recipes' for k in ('path','functionality','direct_matlab_callers','associated_wafo_functions','python_api','python_status','python_limitations'))+' |\n')
    manifest = []
    modules = {}
    adaptations = {'dat2spec','detrendma','spec2sdat','spec2cov','genchol','gaussq','comnsize','kde','mkernel','res2arfc',
                   'dat2crossind','mvnormprb','mvnormpcprb','bvnormprb','bvnormcdf','tranproc','dat2gaus','gaus2dat','duffsim','cc2cmat','hns',
                   'spec2char','dat2cov','dat2cor','rind','mvnortpcprb',
                   'spec2Acdf','spec2mmtpdf','spec2thpdf','spec2tccpdf','wminmax','down2cc','lc2dplus','rindoptset',
                   'dat2tc','dat2wa','dat2midind','findoutliers','hs2sign','vc2sign',
                   'qlevels','qlevels2','gridcount','bincount','parseoptions','gaussq2d','ccquad','smooth','wafoversion'}
    for rel, text in matlab.items():
        path = Path(rel)
        module = path.parts[0] if len(path.parts)>1 else 'root'
        name = path.stem
        comments = [s.strip().lstrip('%').strip() for s in text.splitlines() if s.strip().startswith('%')]
        summary = next((s for s in comments if s and not s.startswith(('CALL:', 'Call:'))), '')
        declaration = next((s.strip() for s in text.splitlines() if s.lstrip().startswith('function')), '')
        commands = [s.split('%',1)[0].strip() for s in text.splitlines() if s.strip() and not s.lstrip().startswith('%')]
        commands = [s.rstrip(';').strip() for s in commands if s]
        help_only = all(re.fullmatch(r'(more\s+(on|off)|help\s+[A-Za-z_][A-Za-z_0-9]*)', s, re.I) for s in commands)
        record_kind = 'function' if declaration else 'documentation' if name.lower() in ('contents','readme') or help_only else 'script'
        status, python = 'pending', ''
        if name.lower() in ('contents', 'readme') or not declaration:
            status = 'documentation-or-script'
        if record_kind != 'documentation':
            if module not in modules:
                try:
                    modules[module] = importlib.import_module('wafo.'+module)
                except ImportError:
                    modules[module] = None
            mod = modules[module]
            if mod is not None and callable(vars(mod).get(name)):
                python = 'wafo.'+module+'.'+name
                status = 'adapted' if name in adaptations or name.endswith(('fit','rnd')) else 'implemented'
                if vars(mod)[name].__module__ in {
                    'wafo.wstats._conditional','wafo.wavemodels._tables',
                    'wafo.multidim._estimation','wafo.wsim._simulation',
                    'wafo.trgauss._transformations','wafo.kdetools._estimation',
                    'wafo.misc._extended','wafo.spec._transforms','wafo.spec._plotting',
                    'wafo.wstats._plotting','wafo.cycles._remaining','wafo.cycles._plotting',
                    'wafo.markov._remaining','wafo.markov._plotting','wafo.damage._remaining',
                    'wafo.trgauss._remaining','wafo.onedim._remaining',
                    'wafo.onedim._plotting','wafo.onedim._reconstruction',
                    'wafo.misc._interactive','wafo.kdetools._sampling','wafo.kdetools._bandwidth','wafo.kdetools._gui',
                    'wafo.spec._directional_transforms','wafo.docs._help','wafo.trgauss._plotting','wafo.trgauss._compatibility',
                    'wafo.wdemos._helpers','wafo.wdemos._demos','wafo.root',
                    'wafo.wavemodels._remaining_models','wafo.cycles._extrapolation',
                    'wafo.wsim._sea','wafo.wsim._cycle_helpers','wafo.wsim._linear_spectrum',
                    'wafo.wavemodels._additional_tables','wafo.markov._estimation',
                    'wafo.trgauss._quadratic_moments','wafo.trgauss._quadratic_geometry','wafo.trgauss._quadratic_saddlepoint','wafo.papers._sessions','wafo.papers',
                    'wafo.multidim._advanced_estimation'} or vars(mod)[name].__module__.startswith('wafo.m_map.'):
                    status = 'adapted'
        if module == 'misc' and name == 'levels':
            python, status = 'wafo.misc.levels', 'implemented'
        manifest.append(dict(matlab=rel, name=name, module=module, declaration=declaration, record_kind=record_kind,
                             functionality=summary, status=status, python=python))
    (ROOT/'src/wafo/conversion_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (docs/'conversion_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    structure = []
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        content = directory/'Contents.m'
        description_ = content.read_text(encoding='latin1').splitlines()[:60] if content.exists() else []
        n = sum(1 for p in directory.rglob('*.m'))
        structure.append(f'## {directory.name} ({n} MATLAB files)\n\n```matlab\n'+ '\n'.join(description_)+'\n```\n')
        target = ROOT/'src/wafo'/directory.name
        target.mkdir(exist_ok=True)
        init = target/'__init__.py'
        if not init.exists():
            init.write_text(f'"""{directory.name}: source namespace reserved; see conversion manifest for pending functions."""\n', encoding='utf-8')
    (docs/'TOOLBOX_STRUCTURE.md').write_text('# Supplied MATLAB toolbox structure\n\n'+ '\n'.join(structure),encoding='utf-8')
    # Installable, local documentation; no dependence on the source checkout
    # or on the original compiled HTML-help viewer.
    help_data = ROOT/'src/wafo/docs/data'
    help_data.mkdir(exist_ok=True)
    topics = {}
    for rel, body in matlab.items():
        if rel.startswith('docs/') and len(Path(rel).parts) == 2 or rel.lower() == 'readme.m' or next(r for r in manifest if r['matlab']==rel)['record_kind']=='documentation':
            comments = [line.lstrip()[1:].lstrip() for line in body.splitlines() if line.lstrip().startswith('%')]
            if comments:
                topics[Path(rel).stem] = '\n'.join(comments)
    (help_data/'topics.json').write_text(json.dumps(topics,indent=2),encoding='utf-8')
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(r[k])+'</td>' for k in ('matlab','functionality','status','python'))+'</tr>' for r in manifest)
    notes = ''.join('<details><summary>'+html.escape(name)+'</summary><pre>'+html.escape(body)+'</pre></details>' for name,body in sorted(topics.items()))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>WAFO Python conversion help</title><style>body{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:1rem}td,th{border-bottom:1px solid #ccc;padding:.4rem;text-align:left}table{border-collapse:collapse}pre{white-space:pre-wrap}summary{cursor:pointer;padding:.5rem}</style><h1>WAFO Python conversion help</h1><p>This is an in-progress conversion of the supplied MATLAB toolbox. Implemented/adapted entries identify Python APIs; their docstrings describe Python arguments and limits. Passing Python tests do not certify MATLAB parity. Original MATLAB topic notes follow the catalog.</p><table><thead><tr><th>Source</th><th>Purpose</th><th>Status</th><th>Python API</th></tr></thead><tbody>'+rows+'</tbody></table><h2>Original source documentation</h2>'+notes+'</html>'
    (help_data/'index.html').write_text(page,encoding='utf-8')
    documented = [r for r in manifest if r['record_kind']=='documentation']
    reviewed = ['# Reviewed source documentation','', 'These files contain only comments or the original MATLAB help/pager commands (plus conventional Contents/Readme files). Their text is bundled in the installed offline help. They contain no numerical workflow requiring a Python script.', '', '| Source | Python access |', '|---|---|']
    reviewed += ['| '+r['matlab']+' | wafo.docs.wafohelp(topic="'+r['name']+'") |' for r in documented if r['name'].lower() not in ('contents','readme')]
    (docs/'REVIEWED_SOURCE_DOCUMENTATION.md').write_text('\n'.join(reviewed)+'\n',encoding='utf-8')
    with (docs/'FUNCTION_CATALOG.md').open('w',encoding='utf-8') as f:
        f.write('# All MATLAB source files and conversion status\n\n`implemented` means a corresponding Python implementation exists; it does not certify numerical parity with MATLAB. `adapted` additionally marks known API/algorithm restrictions. Tests cover representative core workflows; many entries have no individual parity fixture.\n\n')
        f.write('| MATLAB file | Functionality | Status | Python API |\n|---|---|---|---|\n')
        for r in manifest:
            f.write('| '+' | '.join(r[k].replace('|','/') for k in ('matlab','functionality','status','python'))+' |\n')
    data = ROOT/'src/wafo/data'
    for p in (source/'data').iterdir():
        if p.is_file() and p.suffix.lower() in ('.dat','.mat','.asc'):
            shutil.copy2(p, data/p.name)
    counts = {status: sum(r['status']==status for r in manifest) for status in sorted({r['status'] for r in manifest})}
    print(json.dumps(dict(matlab_files=len(manifest), non_matlab_files=len(records), native_files=len(native), coverage=counts),indent=2))


if __name__ == '__main__':
    main()
