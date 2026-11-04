from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal
from wafo.exec.__main__ import main
from wafo.docs import converters


def test_npz_numerical_driver(tmp_path):
    source,output = tmp_path/'in.npz',tmp_path/'out.npz'
    np.savez(source,x=[-1,1,-1],v=0)
    assert main(['findcross',str(source),str(output)])==0
    with np.load(output,allow_pickle=False) as result:
        assert_array_equal(result['result'],[0,1])


def test_npz_rind_driver_returns_numerical_status(tmp_path):
    source,output = tmp_path/'rind.npz',tmp_path/'result.npz'
    np.savez(source,S=np.eye(2),m=[0,0],Blo=-np.inf,Bup=0,indI=[0,2],
             full_output=True,options_json='{"seed": 7, "maxpts": 1024}')
    assert main(['rind',str(source),str(output)])==0
    with np.load(output,allow_pickle=False) as result:
        assert_array_equal(result['result__0'],[.25])
        assert_array_equal(result['result__3__inform'],[0])


def test_npz_wave_driver_accepts_spectrum_fields(tmp_path):
    source,output = tmp_path/'wave.npz',tmp_path/'wave-result.npz'
    w = np.linspace(0,8,101)
    np.savez(source,S__w=w,S__S=np.sqrt(2/np.pi)*np.exp(-w*w/2),
             definition='Tc',paramt=[0,2,3],options_json='{"seed": 7, "maxpts": 1024}')
    assert main(['spec2thpdf',str(source),str(output)])==0
    with np.load(output,allow_pickle=False) as result:
        assert result['result__f'].shape==(3,)
        assert np.all(np.isfinite(result['result__f']))


def test_publication_converters_use_safe_argument_lists(tmp_path,monkeypatch):
    commands = []
    monkeypatch.setattr(converters,'_executable',lambda names,supplied=None:'converter')
    monkeypatch.setattr(converters.subprocess,'run',lambda args,check:commands.append(args))
    fig = tmp_path/'drawing with spaces.fig'
    fig.write_text('#FIG 3.2\n')
    eps,overlay = converters.convert_fig(fig,tmp_path/'output')
    assert commands[0]==['converter','-L','pstex',str(fig.resolve()),str(eps)]
    assert commands[1][-1]==str(overlay)
    ps = tmp_path/'paper.ps'
    ps.write_text('%!PS\n')
    pdf = converters.convert_ps(ps,tmp_path/'output')
    assert commands[2][-1]==str(ps)
    assert '-sOutputFile='+str(pdf) in commands[2]
