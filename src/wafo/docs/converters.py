"""Python replacements for the two publication-conversion C-shell scripts.

Figure and PostScript rendering remains the job of fig2dev and Ghostscript.
These helpers replace hard-coded shell paths/moves with explicit output paths.
They never change or move the original publication files.
"""
from pathlib import Path
import shutil
import subprocess


def _executable(names, supplied=None):
    if supplied is not None:
        path = Path(supplied)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(str(supplied))
        if found:
            return found
        raise FileNotFoundError(f"Converter not found: {supplied}")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError('Install a converter or pass its executable path: '+', '.join(names))


def convert_fig(source, output_dir, *, fig2dev=None):
    """Convert an Xfig .fig to EPS artwork + LaTeX text overlay.

    This refers to Xfig drawings (header #FIG), not MATLAB binary .fig files.
    Returns (eps_path, latex_overlay_path). Requires fig2dev.
    """
    source = Path(source).resolve()
    if not source.is_file() or not source.read_bytes().startswith(b'#FIG'):
        raise ValueError("Expected an existing Xfig drawing beginning with #FIG")
    executable = _executable(('fig2dev',),fig2dev)
    target = Path(output_dir).resolve()
    target.mkdir(parents=True,exist_ok=True)
    eps, overlay = target/(source.stem+'.eps'), target/(source.stem+'.pstex_t')
    subprocess.run([executable,'-L','pstex',str(source),str(eps)],check=True)
    subprocess.run([executable,'-L','pstex_t','-p',eps.name,str(source),str(overlay)],check=True)
    return eps,overlay


def convert_ps(source, output_dir, *, ghostscript=None):
    """Convert PS/EPS to PDF via Ghostscript. Returns the generated PDF path."""
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() not in ('.ps','.eps'):
        raise ValueError("Expected an existing PostScript/EPS file")
    executable = _executable(('gswin64c','gswin32c','gs'),ghostscript)
    target = Path(output_dir).resolve()
    target.mkdir(parents=True,exist_ok=True)
    output = target/(source.stem+'.pdf')
    subprocess.run([executable,'-dBATCH','-dNOPAUSE','-dSAFER','-sDEVICE=pdfwrite',
                    '-sOutputFile='+str(output),str(source)],check=True)
    return output
