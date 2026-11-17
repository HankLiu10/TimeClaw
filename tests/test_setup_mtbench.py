"""Exercise dependency checkout lifecycle with a tiny local Git origin."""
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import setup_mtbench as setup


@pytest.fixture
def local_origin(tmp_path,monkeypatch):
    origin=tmp_path/'origin';subprocess.run(['git','init','--quiet',str(origin)],check=True)
    (origin/'README.md').write_text('pinned dependency fixture\n')
    subprocess.run(['git','-C',str(origin),'add','README.md'],check=True)
    subprocess.run(['git','-C',str(origin),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','--quiet','-m','fixture'],check=True)
    revision=subprocess.check_output(['git','-C',str(origin),'rev-parse','HEAD'],text=True).strip()
    monkeypatch.setattr(setup,'URL',str(origin));monkeypatch.setattr(setup,'REVISION',revision)
    target=tmp_path/'installed';monkeypatch.setattr(sys,'argv',['setup','--destination',str(target)])
    return target,revision


def test_clean_pinned_checkout_is_installed_and_reused(local_origin):
    target,revision=local_origin
    setup.main();setup.main()
    assert subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()==revision
    assert (target/'README.md').read_text()=='pinned dependency fixture\n'


def test_dirty_existing_dependency_is_not_overwritten(local_origin):
    target,_=local_origin;setup.main()
    (target/'README.md').write_text('keep user changes\n')
    with pytest.raises(RuntimeError,match='local changes'):setup.main()
    assert (target/'README.md').read_text()=='keep user changes\n'


def test_failed_fetch_does_not_leave_an_unusable_installation(local_origin,monkeypatch):
    target,_=local_origin;original_run=subprocess.run
    def fetch_failure(command,*args,**kwargs):
        if 'fetch' in command:raise subprocess.CalledProcessError(128,command)
        return original_run(command,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(setup.subprocess,'run',fetch_failure)
        with pytest.raises(subprocess.CalledProcessError):setup.main()
    assert not target.exists(),'A failed fetch must not publish a partial final destination'
    setup.main()
    assert (target/'README.md').is_file()
