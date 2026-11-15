"""Standalone necessary CPU arithmetic preconditions; not a program proof.

Only this worker's Torch thread/matmul settings are changed. MXCSR and environment
variables are read, never modified. The helper executes only register/control
reads; compilation artifacts live in a fresh /tmp directory.
"""
import argparse
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

C_SOURCE = r'''#include <xmmintrin.h>
#include <fenv.h>
unsigned int guard_read_mxcsr(void) { return _mm_getcsr(); }
int guard_read_round(void) { return fegetround(); }
int guard_nearest_constant(void) { return FE_TONEAREST; }
'''


def check_control_values(mxcsr, rounding, nearest):
    """Pure mask validation, also used for adverse synthetic control states."""
    checks = {
        'mxcsr_round_nearest_even': ((mxcsr >> 13) & 3) == 0,
        'mxcsr_daz_disabled': (mxcsr & (1 << 6)) == 0,
        'mxcsr_ftz_disabled': (mxcsr & (1 << 15)) == 0,
        'fegetround_nearest': rounding == nearest,
    }
    return {'checks': checks, 'passed': all(checks.values())}


def mask_unit_cases():
    normal = 0x1f80
    assert check_control_values(normal, 0, 0)['passed']
    for mode in (1, 2, 3):
        result = check_control_values(normal | (mode << 13), 0, 0)
        assert not result['passed'] and not result['checks']['mxcsr_round_nearest_even']
    for bit, check in ((6, 'mxcsr_daz_disabled'), (15, 'mxcsr_ftz_disabled')):
        result = check_control_values(normal | (1 << bit), 0, 0)
        assert not result['passed'] and not result['checks'][check]
    assert not check_control_values(normal, 1, 0)['passed']
    return {'passed': True, 'case_count': 7, 'scope': 'synthetic flag masks; no control register writes'}


def probe():
    report = {
        'schema_version': 'ares.cpu_arithmetic_guard.v1',
        'cpu_preconditions_met': False,
        'full_program_arithmetic_proved': False,
        'flight_certificate': False,
        'host': {'machine': platform.machine(), 'platform': platform.platform()},
        'c_source': C_SOURCE,
        'c_source_sha256': hashlib.sha256(C_SOURCE.encode()).hexdigest(),
        'mask_unit_cases': mask_unit_cases(),
        'execution': {'device': 'cpu', 'eager_only': True, 'cuda_used': False,
                      'torch_compile_used': False, 'mxcsr_modified': False,
                      'environment_flags_modified': False},
    }
    if platform.machine().lower() not in ('x86_64', 'amd64', 'i386', 'i686'):
        report['failure'] = 'unsupported architecture: x86 MXCSR helper required'
        return report
    compiler = shutil.which('cc') or shutil.which('gcc')
    if compiler is None:
        report['failure'] = 'C compiler unavailable'
        return report
    work = Path(tempfile.mkdtemp(prefix='ares-cpu-arithmetic-guard-'))
    source, library = work / 'guard.c', work / 'guard.so'
    source.write_text(C_SOURCE)
    command = [compiler, '-shared', '-fPIC', '-O2', '-msse2', str(source), '-lm', '-o', str(library)]
    report['helper'] = {'compiler': compiler, 'command': command,
                        'compiler_version': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                        'source_path': str(source), 'library_path': str(library)}
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        report['helper']['compile_stderr'] = result.stderr
        report['helper']['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
        helper = ctypes.CDLL(str(library))
        helper.guard_read_mxcsr.restype = ctypes.c_uint
        helper.guard_read_round.restype = ctypes.c_int
        helper.guard_nearest_constant.restype = ctypes.c_int
        import torch
        # These process-local settings are necessary to the declared CPU worker.
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        mxcsr = helper.guard_read_mxcsr()
        rounding = helper.guard_read_round()
        nearest = helper.guard_nearest_constant()
        controls = check_control_values(mxcsr, rounding, nearest)
        tensor = torch.tensor([1.0], dtype=torch.float32, device='cpu')
        checks = {**controls['checks'],
                  'torch_explicit_cpu_float32': tensor.device.type == 'cpu' and tensor.dtype == torch.float32,
                  'torch_default_float32': torch.get_default_dtype() == torch.float32,
                  'torch_matmul_highest': torch.get_float32_matmul_precision() == 'highest',
                  'torch_single_thread': torch.get_num_threads() == 1}
        report['torch'] = {'version': torch.__version__, 'git_version': torch.version.git_version,
                           'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                           'default_dtype': str(torch.get_default_dtype()),
                           'probe_dtype': str(tensor.dtype), 'num_threads': torch.get_num_threads(),
                           'matmul_precision': torch.get_float32_matmul_precision()}
        report['control_state'] = {'mxcsr': mxcsr, 'mxcsr_hex': hex(mxcsr),
                                   'fegetround': rounding, 'FE_TONEAREST': nearest}
        report['checks'] = checks
        report['cpu_preconditions_met'] = all(checks.values())
        report['scope_limit'] = ('Current standalone worker thread only; not an audit of every backend/library thread, '
                                 'all compiler transformations, transcendental error bounds, or the full program. '
                                 'Caller computations must remain CPU/eager/Float32 and preserve these controls.')
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = probe()
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')
    raise SystemExit(0 if report['cpu_preconditions_met'] else 1)
