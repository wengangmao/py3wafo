"""Source-formula and independent rainflow-count checks for FAT helpers."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from wafo.wsim import fr2nt,fr2comb,comb2pro,getulcc,fr2res,rfc2load_fat
from wafo.cycles import tp2rfc


def fat_count(x,n,definition='CS',closed=False):
    """Count actual turning points, then map minimum/maximum to FAT columns."""
    cycles=tp2rfc(x,definition,full_output=True)[1 if closed else 0]
    result=np.zeros((n,n))
    if len(cycles):
        pairs=cycles.astype(int)
        np.add.at(result,(pairs[:,0],n-1-pairs[:,1]),1)
    return result


def frequency(n=5):
    f=np.zeros((n,n))
    f[0,0]=2
    f[1,1]=4
    return f


def test_counting_distribution_literal_index_formula():
    f=np.random.default_rng(3).integers(0,5,(6,6))
    expected=np.zeros_like(f)
    for i in range(1,5):
        for j in range(1,5):
            if i+j<=5:
                expected[i,j]=sum(f[p,q] for p in range(i) for q in range(j))
    assert_array_equal(fr2nt(f),expected)
    F=np.zeros((3,3));F[0,0]=7
    assert_array_equal(fr2nt(F),[[0,0,0],[0,7,0],[0,0,0]])
    for invalid in (np.zeros((2,2)),np.zeros((3,4)),np.full((3,3),-1),np.full((3,3),np.nan)):
        with pytest.raises(ValueError):fr2nt(invalid)


def test_combinations_preserve_source_cartesian_residual_updates():
    f=frequency(6)
    r=np.array([5,0,4,1])
    expected=np.zeros_like(f)
    nt=fr2nt(f)
    for i in range(6):
        for j in range(6-i):
            expected[i,j]=2*nt[i,j]+sum(f[:i,j])+sum(f[i,:j])
    for a,b in zip(r[:-1],r[1:]):
        indices=range(b+1,a+1) if a>b else range(a,b)
        for row in indices:
            for col_level in indices:
                expected[row,5-col_level]+=1
    expected[np.indices(f.shape).sum(axis=0)>=5]=0
    actual=fr2comb(f,r)
    assert_array_equal(actual,expected)
    assert np.any(fr2comb(np.zeros((6,6)),[5,0])>0)
    assert_array_equal(f,frequency(6))
    with pytest.raises(ValueError):fr2comb(f,[6,0])


def test_placement_probability_exact_formula_and_zero_cases():
    c=np.array([3,5,2.]);f=np.array([2,3,1.])
    p=(1-1/c)**f
    expected=np.r_[np.prod(p),1-p[0],(1-p[1])*p[0],(1-p[2])*p[0]*p[1]]
    actual=comb2pro(c[:,None],f)
    assert_allclose(actual,expected,atol=1e-15)
    assert_allclose(sum(actual),1,atol=1e-15)
    assert_array_equal(comb2pro([0,0,1],[0,1,0]),[0,0,1,0])
    assert_array_equal(comb2pro([],[]),[1])
    assert_array_equal(comb2pro([0,1,5],[0,0,0]),[1,0,0,0])
    with pytest.raises(ValueError):comb2pro([0],[2])
    with pytest.raises(ValueError):comb2pro([1,2],[3])


def test_upper_left_search_zero_based_ties_and_empty():
    f=frequency()
    assert getulcc(f)==(0,0,2)
    f[0,0]=0
    assert getulcc(f)==(1,1,4)
    f[0,2]=1
    with pytest.raises(ValueError,match='not unique'):getulcc(f)
    with pytest.raises(ValueError,match='No positive'):getulcc(np.zeros((5,5)))
    assert getulcc([[3]])==(0,0,3)


def test_residual_fast_path_and_seeded_source_walk():
    f=np.zeros((4,4));f[0,0]=1
    assert_array_equal(fr2res(f,seed=4),[3,0])
    f=frequency()
    a=fr2res(f,seed=12,burn_in=30,max_steps=10000)
    b=fr2res(f,seed=12,burn_in=30,max_steps=10000)
    assert_array_equal(a,b)
    assert len(a)>2 and np.all((a>=0)&(a<5))
    assert_array_equal(f,frequency())
    assert np.all(np.diff(a)!=0)
    assert_array_equal(fr2res(f,seed=np.random.default_rng(12),burn_in=30),a)


def test_residual_nonterminating_source_case_has_explicit_bound():
    f=np.zeros((6,6));f[0,0]=1
    with pytest.raises(RuntimeError,match='max_steps'):
        fr2res(f,seed=2,max_steps=30)
    with pytest.raises(ValueError):fr2res(np.eye(5))
    with pytest.raises(ValueError):fr2res(f,burn_in=0)


@pytest.mark.parametrize('seed',[0,1,2,3,10,20])
def test_load_exact_closed_cycle_counts_across_random_placements(seed):
    f=frequency()
    x,res,comb,remaining=rfc2load_fat(f,[4,0],seed=seed,full_output=True)
    assert_array_equal(res,[4,0])
    assert_array_equal(remaining,0)
    assert_array_equal(comb,0)
    assert_array_equal(fat_count(x,5,closed=True),f)
    assert_array_equal(fat_count(x,5),f+fat_count(res,5))
    assert np.all(np.diff(x)[:-1]*np.diff(x)[1:]<0)
    assert_array_equal(x,rfc2load_fat(f,[4,0],seed=seed))
    assert_array_equal(f,frequency())


def test_load_auto_residual_and_source_fractional_scaling():
    f=np.zeros((3,3));f[0,0]=4
    x,r,_,remaining=rfc2load_fat(f,seed=1,full_output=True)
    assert_array_equal(fat_count(x,3),f)
    assert_array_equal(r,[2,0])
    assert_array_equal(remaining,0)
    fractional=frequency()/100
    with pytest.warns(RuntimeWarning,match='floor'):
        x=rfc2load_fat(fractional,[4,0],seed=3)
    assert_array_equal(fat_count(x,5,closed=True),frequency())


def test_expected_cycle_stationary_scaling_and_work_bound():
    f=frequency()
    x,r,c,remaining=rfc2load_fat(f,[4,0],20,seed=9,full_output=True)
    assert len(x)>44 and len(x)<=46
    assert remaining.sum()>999000
    assert np.all(remaining>=0) and np.all((x>=0)&(x<5))
    assert_array_equal(x,rfc2load_fat(f,[4,0],20,seed=9))
    with pytest.raises(RuntimeError,match='max_steps'):
        rfc2load_fat(f,[4,0],20,seed=9,max_steps=3)
    with pytest.raises(ValueError,match='exhausted'):
        rfc2load_fat(f,[1,0],seed=2)
    with pytest.raises(ValueError):rfc2load_fat(f,[4,0],num_cc=-1)
    with pytest.raises(ValueError):rfc2load_fat(f,[4,0,1,2])
