import py3wafo
import wafo
from py3wafo.trgauss import rind
from py3wafo.spec import jonswap
from numpy.testing import assert_allclose


def test_requested_package_name_preserves_implementation_identity():
    assert py3wafo.spec is wafo.spec
    assert rind is wafo.trgauss.rind
    assert jonswap is wafo.spec.jonswap
    assert_allclose(py3wafo.spec.spec2mom(jonswap(sdata=[4,8]),0),[1])
    assert py3wafo.coverage()==wafo.coverage()
