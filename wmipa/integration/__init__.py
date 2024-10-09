from shutil import which

import os

# TODO: Hack!
# export PATH=/afs/inf.ed.ac.uk/user/s26/s2626206/.wmipa/latte/bin:$PATH
os.environ['PATH'] = '/afs/inf.ed.ac.uk/user/s26/s2626206/.wmipa/latte/bin:' + os.environ['PATH']
# export PATH=/afs/inf.ed.ac.uk/user/s26/s2626206/.wmipa/approximate-integration/bin:$PATH
os.environ['PATH'] = '/afs/inf.ed.ac.uk/user/s26/s2626206/.wmipa/approximate-integration/bin:' + os.environ['PATH']


def _is_latte_installed():
    return which("integrate") is not None


def _is_volesti_installed():
    return which("volesti_integrate") is not None


def _is_symbolic_installed():
    try:
        from pywmi.engines import PyXaddEngine
        return True
    except ImportError:
        return False


IMPORT_ERR_MSG = "No integration backend installed. Run `wmipa-install --help` for more information."

if not any((_is_latte_installed(), _is_volesti_installed(), _is_symbolic_installed())):
    raise ImportError(IMPORT_ERR_MSG)
else:
    from .latte_integrator import LatteIntegrator
    from .symbolic_integrator import SymbolicIntegrator
    from .volesti_integrator import VolestiIntegrator
