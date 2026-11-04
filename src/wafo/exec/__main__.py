"""Portable numerical driver: python -m wafo.exec ROUTINE input.npz output.npz.

NPZ input keys are Python keyword parameter names. Optional `options_json`
contains a JSON object for a routine's options. No pickled objects are loaded.
"""
import argparse
import importlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
from pathlib import Path
import numpy as np


ROUTINES = {
    'findcross': 'wafo.onedim.findcross',
    'findrfc': 'wafo.onedim.findrfc',
    'csort': 'wafo.misc.csort',
    'disufq': 'wafo.wsim.disufq',
    'simduff': 'wafo.wsim.simduff',
    'rind': 'wafo.trgauss.rind',
    'mvnormprb': 'wafo.trgauss.mvnormprb',
    'mvnortpcprb': 'wafo.trgauss.mvnortpcprb',
    'mvtprb': 'wafo.trgauss.mvtprb',
    'spec2Acdf': 'wafo.trgauss.spec2Acdf',
    'spec2mmtpdf': 'wafo.trgauss.spec2mmtpdf',
    'spec2thpdf': 'wafo.trgauss.spec2thpdf',
    'spec2tccpdf': 'wafo.trgauss.spec2tccpdf',
    'wminmax': 'wafo.trgauss.wminmax',
    'down2cc': 'wafo.damage.down2cc',
}


def _flatten(value, prefix='result'):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        result = {}
        for key,item in value.items():
            result.update(_flatten(item,prefix+'__'+str(key)))
        return result
    if isinstance(value, (tuple,list)):
        result = {}
        for i,item in enumerate(value):
            result.update(_flatten(item,prefix+'__'+str(i)))
        return result
    if value is None:
        return {prefix: np.asarray('None')}
    a = np.asarray(value)
    if a.dtype == object:
        raise TypeError(f"Cannot serialize {prefix} without pickling")
    return {prefix:a}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('routine', choices=sorted(ROUTINES))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    module,name = ROUTINES[args.routine].rsplit('.',1)
    function = getattr(importlib.import_module(module),name)
    with np.load(args.input,allow_pickle=False) as archive:
        values = {key:(archive[key].item() if archive[key].ndim==0 else archive[key]) for key in archive.files}
    if 'options_json' in values:
        values['options'] = json.loads(values.pop('options_json'))
    # Spectrum structs are carried as S__w, S__S, S__h NPZ keys.
    spectrum = {key[3:]:values.pop(key) for key in list(values) if key.startswith('S__')}
    if spectrum:
        values['S' if 'S' in inspect.signature(function).parameters else 'spec'] = spectrum
    result = function(**values)
    with args.output.open('wb') as out:
        np.savez_compressed(out, **_flatten(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
