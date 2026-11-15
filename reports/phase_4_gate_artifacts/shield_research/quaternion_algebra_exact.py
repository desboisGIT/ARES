"""Exact polynomial identities for the vendored real quaternion expressions.

These justify algebraic interval-analysis rewrites. They do not justify binary
floating reassociation, a nonlinear invariant set, or flight certification.
"""
from pathlib import Path
import hashlib
import json
import sympy as S


def multiply_vendor(a,b):
    x1,y1,z1,w1=a;x2,y2,z2,w2=b
    ww=(z1+x1)*(x2+y2);yy=(w1-y1)*(w2+z2);zz=(w1+y1)*(w2-z2)
    xx=ww+yy+zz;qq=(xx+(z1-x1)*(x2-y2))/2
    return S.Matrix([qq-xx+(x1+w1)*(x2+w2),qq-yy+(w1-x1)*(y2+z2),
                     qq-zz+(z1+y1)*(w2-x2),qq-ww+(z1-y1)*(y2-z2)])


def multiply_standard(a,b):
    av,bv=S.Matrix(a[:3]),S.Matrix(b[:3]);aw,bw=a[3],b[3]
    xyz=aw*bv+bw*av+av.cross(bv)
    return S.Matrix([*xyz,aw*bw-av.dot(bv)])


def rotation(q):
    x,y,z,w=q;v=S.Matrix([x,y,z])
    cross=S.Matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (w*w-v.dot(v))*S.eye(3)+2*w*cross+2*v*v.T


def zero(matrix):
    return all(S.expand(x)==0 for x in matrix)


if __name__=='__main__':
    a=S.symbols('ax ay az aw',real=True);b=S.symbols('bx by bz bw',real=True)
    ra,rb=rotation(a),rotation(b);ab=multiply_vendor(a,b)
    norm2=sum(x*x for x in a)
    result={'schema_version':'ares.quaternion_real_algebra.v1',
        'vendor_mul_equals_Hamilton':zero(ab-multiply_standard(a,b)),
        'rotation_homomorphism_without_unit_assumption':zero(rotation(ab)-ra*rb),
        'isotropic_drag_homogeneous_norm_identity':zero(ra*ra.T-norm2**2*S.eye(3)),
        'quaternion_norm_multiplicative':S.expand(sum(x*x for x in ab)-norm2*sum(x*x for x in b))==0,
        'sympy':S.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'vendor_math_sha256':hashlib.sha256(Path('ares/vendor/diffaero/math.py').read_bytes()).hexdigest(),
        'real_arithmetic_only':True,'floating_reassociation_proved':False,
        'flight_certificate':False}
    assert all(result[key] for key in ('vendor_mul_equals_Hamilton',
        'rotation_homomorphism_without_unit_assumption','isotropic_drag_homogeneous_norm_identity',
        'quaternion_norm_multiplicative'))
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
