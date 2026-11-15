"""Binary32 sqrt with exact rational rounding-bin acceptance.

math.sqrt/Float64 is only a candidate guess. Every returned finite word has an
exact squared-bin witness, including midpoint ties and subnormal endpoints.
"""
from fractions import Fraction as F
import math,struct

def bits(x):return struct.unpack('>I',struct.pack('>f',float(x)))[0]
def word(index):return struct.unpack('>f',struct.pack('>I',index))[0]
def rational(x):
 if isinstance(x,F):return x
 if isinstance(x,bool) or not math.isfinite(float(x)):raise ValueError('finite nonnegative argument required')
 return F.from_float(float(x))
MAX=F.from_float(word(0x7f7fffff))
def sqrt_fraction(value):
 """Exact rational extension, useful for tie proofs; finite output only."""
 x=rational(value)
 if not 0<=x<=MAX:raise ValueError('nonnegative finite binary32-range input required')
 candidate=bits(math.sqrt(float(x)))
 for index in range(max(0,candidate-2),candidate+3):
  center=F.from_float(word(index));before=-F.from_float(word(1)) if index==0 else F.from_float(word(index-1));after=F.from_float(word(index+1))
  lo=(before+center)/2;hi=(center+after)/2;even=index%2==0
  low_ok=lo<0 or x>lo*lo or even and x==lo*lo
  high_ok=x<hi*hi or even and x==hi*hi
  if low_ok and high_ok:
   return word(index),dict(input_exact=x,output_bits=index,rounding_bin_lower=lo,rounding_bin_upper=hi,lower_midpoint_tie=lo>=0 and x==lo*lo,upper_midpoint_tie=x==hi*hi,ties_to_even=even,exact_squared_bin_pass=True,flight_certificate=False)
 raise ArithmeticError('no exact sqrt rounding bin isolated; candidate is never trusted')
def sqrt_word(value):
 """Only represented nonnegative finite binary32 inputs are accepted."""
 x=rational(value)
 if not 0<=x<=MAX or F.from_float(word(bits(x)))!=x:raise ValueError('argument must be an exact finite nonnegative binary32 word')
 if bits(value)==0x80000000:
  return -0.,dict(input_exact=F(0),output_bits=0x80000000,exact_squared_bin_pass=True,signed_zero_preserved=True,flight_certificate=False)
 return sqrt_fraction(x)
