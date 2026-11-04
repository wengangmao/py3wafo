"""Source-family associations for generated inventories (not binary emulation)."""


def replacement_for(path):
    name = path.replace('\\','/').rsplit('/',1)[-1].split('.',1)[0].lower()
    direct = {
        'findcross': ('wafo.onedim.findcross','C level-crossing algorithm; zero-based indices'),
        'findrfc': ('wafo.onedim.findrfc','C rainflow-filter algorithm with terminal zero padding'),
        'csort': ('wafo.misc.csort','Stable integer sorting; NumPy stable sort replaces counting sort'),
        'disufq': ('wafo.wsim.disufq','C finite/deep-water interaction formulas; resonances explicitly rejected'),
        'simduff': ('wafo.wsim.simduff','Fortran Euler/RNG formulas; double precision and explicit integer seed'),
    }
    if name in direct:
        api,notes = direct[name]
        return dict(python_api=api,python_status='ported',python_limitations=notes)
    if name=='down2cc':
        api,status,notes = ('wafo.damage.down2cc','reconstructed',
            'Source is absent: midpoint superlevel-set slicing; not executable parity or a certified finite-grid damage bound')
    elif name.startswith('sp2acdf'):
        api,status,notes = ('wafo.trgauss.spec2Acdf','capability-replacement-limited',
            'Rice integral for time-domain Gaussian crest/trough amplitudes; finite time/barrier grids and numerical integration errors')
    elif name.startswith('sp2mmt') or name.startswith('sp2mm'):
        api,status,notes = ('wafo.trgauss.spec2mmtpdf','capability-replacement-limited',
            'Gaussian consecutive extrema/period density; finite-grid Rice approximation; legacy transforms and rainflow definitions unsupported')
    elif name.startswith('sp2tccpdf'):
        api,status,notes = ('wafo.trgauss.spec2tccpdf','capability-replacement-limited',
            'Gaussian crest-to-crest period density; time-domain subset with finite barrier grids')
    elif name.startswith('sp2thpdf'):
        api,status,notes = ('wafo.trgauss.spec2thpdf','capability-replacement-limited',
            'Gaussian period/amplitude Rice density; selected time-domain definitions only')
    elif name in ('minmax','twog'):
        api,status,notes = ('wafo.trgauss.wminmax','capability-replacement-limited',
            'Gaussian extrema density via Rice integral and sign symmetry; old executable file protocol not reproduced')
    elif name=='mvbrvmod':
        api,status,notes = ('wafo.source.integrate_box','algorithm-replacement',
            'Vector cubature with scrambled Sobol replicas, replacing Korobov/Richtmeyer rules; finite boxes only')
    elif name=='intmodule':
        api,status,notes = ('wafo.source.integrate_box','algorithm-replacement',
            'General cubature consolidated here; RIND-specific truncation/conditioning implemented in wafo.trgauss.rind')
    elif any(s in name for s in ('rind','alanpab','jacobmod')):
        api,status,notes = ('wafo.trgauss.rind','algorithm-replacement',
            'Conditional normal truncated expectation with absolute derivative product; new randomized QMC engine, not old algorithm-selector parity')
    elif 'mvnprodcorr' in name:
        api,status,notes = ('wafo.trgauss.mvnormpcprb','algorithm-replacement',
            'Normal rectangle probability with product correlations; modern backend replaces legacy MEX')
    elif 'mvnprdmex' in name:
        api,status,notes = ('wafo.trgauss.mvnortpcprb','algorithm-replacement',
            'Normal/noncentral Student-t product-correlation probability; modern randomized integration')
    elif name=='mvdist' or name.startswith('mexmvt'):
        api,status,notes = ('wafo.trgauss.mvtprb','algorithm-replacement',
            'Noncentral t/normal rectangle probability with linear constraints; no packed-MEX calling convention')
    elif name=='mulnormex' or name.startswith(('mexmvn','mexgenzmvn')):
        api,status,notes = ('wafo.trgauss.mvnormprb','algorithm-replacement',
            'Normal rectangle probability; modern integration replaces AS195/Genz variants and missing IMSL auxiliary routines')
    else:
        return dict(python_api='',python_status='unmapped',python_limitations='Needs individual source inspection')
    return dict(python_api=api,python_status=status,python_limitations=notes)
