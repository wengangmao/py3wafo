"""papers: reproductions of the WAFO (ISOPE 2000) and reconstruction (ISOPE 1999) papers.

wafodemo/recdemo start an independent session with an optional menu;
wafofig(ix)/recfig(ix) and the numbered wafofigN/recfigN render single
figures and return PaperFigure results. State lives in a PaperSession
instead of MATLAB globals; pass the same session to reuse calculations.
"""
from ._sessions import (PaperFigure, PaperSession, wafointro, recintro, wafoinit, recinit,
    wafocleanup, recleanup, wafofig, recfig, wafodemo, recdemo, papermenu)


def _numbered(paper, number):
    dispatch = wafofig if paper == 'wafo' else recfig

    def figure(session=None, *, strict=False, **settings):
        return dispatch(number, session, strict=strict, **settings)
    figure.__name__ = f'{paper}fig{number}'
    figure.__qualname__ = figure.__name__
    figure.__doc__ = (f'Render {paper}fig{number}: {paper}fig({number}, session, **settings).\n\n'
                      'Returns a PaperFigure; pass a session to reuse cached calculations.')
    return figure


for _number in range(1, 11):
    globals()[f'wafofig{_number}'] = _numbered('wafo', _number)
for _number in range(1, 14):
    globals()[f'recfig{_number}'] = _numbered('rec', _number)
del _number
