import pytest
from wafo import source
from wafo.native import native_files, native_info, resolve_native


def test_every_native_file_has_a_real_capability_entry():
    records = native_files()
    assert len(records)==121
    assert len({row['path'] for row in records})==121
    assert all(row['python_status']!='unmapped' for row in records)
    for row in records:
        assert callable(resolve_native(row['path']))
        assert row['python_limitations']


def test_original_source_and_binary_resolve_to_same_kernel():
    assert resolve_native('source/findcross.c') is source.findcross
    assert resolve_native('findrfc.dll') is source.findrfc
    assert resolve_native('exec/pcwin64/simduff.exe') is source.simduff
    assert resolve_native('mvnprdmex53.dll.dll') is source.mvnortpcprb


def test_reconstruction_and_restricted_driver_are_not_reported_as_parity():
    assert native_info('down2cc.exe')[0]['python_status']=='reconstructed'
    assert native_info('sp2acdf70.exe')[0]['python_status']=='capability-replacement-limited'
    assert native_files('reconstructed')
    with pytest.raises(KeyError):
        resolve_native('not_a_native_file')
