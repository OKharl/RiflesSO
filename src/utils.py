# =================================================================================================
# Utils.py: auxiliary classes and functions for RiflesSO
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Sequence, List, Union, Tuple, IO, BinaryIO, Set, Optional
from numpy.typing import NDArray
import numpy as np
from math import gcd, floor
from fractions import Fraction
from functools import wraps
from time import perf_counter
import re

from . import units

# ------------------------------------------------------------------------------------------------
# General-purpose functions
# ------------------------------------------------------------------------------------------------

def timeit(func):
    '''A decorator to time execution of a function on every call to it. 
    Place right before the function's def.
    '''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'timeit(): call to {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def nparr2str(a: NDArray, prec=3, fmt='fixed', **kwargs):
    "Just an alias for numpy array fomatting: nparr2str(np_array [, prec=3 [, fmt='fixed' [, kwargs]]])"
    if fmt == 'fixed':
        fmt_func = lambda x: format(x, f'.{prec}f')
    elif fmt == 'exponential':
        fmt_func = lambda x: format(x, f'.{prec}e')
    else:
        fmt_func = None
    if fmt_func is None:
        return np.array2string(a, precision=prec, max_line_width=256, *kwargs)
    else:
        return np.array2string(a, max_line_width=256, 
                               formatter={'float_kind': fmt_func, 'complex_kind': fmt_func}, *kwargs
                              )

def write_array_binary(a: NDArray, f: BinaryIO, *, dtype: Optional[np.dtype] = None):
    '''Write an n-dimensional numpy array `a` to a binary file `f`, saving its shape as well.
    Each array element is written as an entity of type `dtype` (if the latter is not specified,
    `a.dtype` is used instead).
    '''
    t = a.dtype if dtype is None else dtype
    dim_shape = np.hstack((a.ndim, a.shape)).astype(np.int32) # Prepended by number of dimensions
    f.write(dim_shape.tobytes())
    if t == a.dtype:
        f.write(a.tobytes(order='C'))
    else:
        f.write(a.astype(t).tobytes(order='C'))

def read_array_binary(f: BinaryIO, dtype: np.dtype) -> NDArray:
    '''Read an n-dimensional numpy array previously written using write_array_binary() 
    from a binary file `f`, automatically reshaping it as required.
    '''
    dim = np.frombuffer(f.read(4), dtype=np.int32, count=1)[0]
    shape = np.frombuffer(f.read(4 * dim), dtype=np.int32, count=dim)
    return np.frombuffer(f.read(np.product(shape) * dtype().itemsize), dtype=dtype).reshape(shape)

# ------------------------------------------------------------------------------------------------
# Regular expression/parsing related functions (to be refactored to a separate parser module)
# ------------------------------------------------------------------------------------------------

# RegEx constants and the corresponding
def _re_group(re: str) -> str: return '(' + re + ')'
def _re_anongroup(re: str) -> str: return '(?:' + re + ')'
_re_int_bare = r'[+-]?\d+'
re_int = _re_anongroup(_re_int_bare)
re_int_group = _re_group(re_int)
_re_real_fixed_bare = _re_int_bare + r'(?:\.\d*)?' + '|' + r'\.\d+'
re_real_fixed = _re_anongroup(_re_real_fixed_bare)
re_real_fixed_group = _re_group(_re_real_fixed_bare)
_re_real_bare = re_real_fixed + _re_anongroup('[eE]' + _re_int_bare) + '?'
re_real_bare = _re_anongroup(_re_real_bare)
re_real_group = _re_group(_re_real_bare)
re_whitespace = r'\s+'
re_opt_whitespace = r'\s*'

def re_multi(re_value: str, nvalues: int, re_sep: str = re_whitespace) -> str:
    'Represents `nvalues` tokens matching `re_value` and separated by `re_sep`'
    return re_sep.join([re_value] * nvalues)

def re_separator(sep_char: str = '-', nmin: int = 1) -> str:
    'Represents a separator made of at least `nmin` atcharacters `sep_char`'
    return '(?:\\' + sep_char + r'){' + str(nmin) + r',}'

def find_line_in_file(f: IO, accepted_line_sequences: List[List[str]], flags=0):
    '''Read input stream `f` until a string or a consecutive group of strings is encountered.
    `accepted_line_sequences`[n][0:-1] is the nth accepted sequence of consecutive strings;
    each of them can be a regular expression with capturing groups.
    Returns a tuple (`n`, [group1, group2, ..., groupk]) corresponding to the line(s) found, 
    or `None` if nothing was found.
    '''
    nseq = len(accepted_line_sequences)
    captured_groups = [[]] * nseq   # No matching lines in each of possible line sequences
    while True:
        ln = f.readline()
        if ln == '':
            return None
        ln = ln.strip()
        for iseq in range(nseq):
            m = re.match(accepted_line_sequences[iseq][len(captured_groups[iseq])], ln, flags)
            if m == None:
                captured_groups[iseq] = []   # Clear matches of previous lines, if any
            else:
                captured_groups[iseq].append(m.groups())
                if len(captured_groups[iseq]) == len(accepted_line_sequences[iseq]):
                    return (iseq, captured_groups[iseq])

# ------------------------------------------------------------------------------------------------
# Units-related functions
# ------------------------------------------------------------------------------------------------
_all_units = { 
                'cm': units.cm, 'Angstrom': units.Angstrom, 'Bohr': units.BohrRadius, 'm': units.meter,
                'g': units.gram, 'kg': units.kg, 'amu': units.amu,
                'sec': units.sec,
                'erg': units.erg, 'J': units.Joule,
                'eV': units.eV, 'meV': units.meV, 'Ha': units.Hartree, 'Ry': 0.5 * units.Hartree,
                'deg': units.Degree, 'rad': 1.0
             }

#def unit_value(unit_spec: str, allowed_units: Optional[Set[str]]=_all_units.keys()) -> float:
#    i = 0
#    
#    def read_token() -> str:
#        while i < len(unit_spec) and unit_spec[i].isspace():
#            i += 1
#        if i >= len(unit_spec):
#            return None
#        tok = ''
#        ch = unit_spec[i]
#        if ch in ['/', '*', '^']:   # A sign
#            i += 1
#            if i < len(unit_spec) and unit_spec[i] == '*':    # A ** (raising to a power) token
#                i += 1
#                ch += '*'
#            return ch
#        elif not ch.isalpha():      # Should be a unit name
#            raise ValueError(f"In unit_value(): unallowed character '{ch}' in unit specification")
#        else:
#            i += 1
#            while i < len(unit_spec) and unit_spec[i].isalnum():
#                ch += unit_spec[i]
#                i += 1
#            return ch
#    
#    valstack = []
#    while 

def float_with_units(s: str, 
                     default_units: Optional[Union[float,str]]=1.0, 
                     allowed_units: Optional[Set[str]]=_all_units.keys()) -> float:
    'Parse a floating-point number followed by one of the allowed_units'
    default_unit_value = _all_units[default_units] if isinstance(default_units, str) else default_units
    m = re.match(re_real_group + '$', s.strip())
    if m != None:
        return float(s) * default_unit_value
    m = re.match(re_real_group + re_whitespace + r'([\w\s\*\/\^]+)' + '$', s.strip())
    if m != None:
        number = float(m.group(1))
        units_spec = m.group(2)
        units_spec = units_spec.replace('^', '**')
        for w in allowed_units:
            units_spec = re.sub(r'(^|[^\w])' + w + r'([^\w]|$)', 
                                lambda m: m.group(1) + '(' + str(_all_units[w]) + ')' + m.group(2), 
                                units_spec)
        try:
            units_value = eval(units_spec, {}, {})
            return number * units_value
        except:
            raise ValueError('In float_with_units(): cannot compile unit specification')


# ------------------------------------------------------------------------------------------------
# Mathematical/linear algebra functions
# ------------------------------------------------------------------------------------------------

def normalize(vec: NDArray) -> NDArray:
    "Normalize an n-dimensional vector to unity "
    return vec / np.linalg.norm(vec)

def plane_vectors(n: NDArray[np.float]) -> Tuple[NDArray[np.float], NDArray[np.float]]:
    "For a normal vector `n` to a plane, return two orthonormal vectors (a, b) in this plane."
    evals, evecs = np.linalg.eigh(np.outer(normalize(n), normalize(n)))
    a, b = evecs[:, 0], evecs[:, 1]
    return (a, b) if np.cross(a, b).dot(n) > 0 else (b, a)

def gram_schmidt(vectors: Sequence[NDArray[np.float]]) -> List[NDArray[np.float]]:
    '''Gram-Schmidt orthogonalization of a list of linearly independent vectors v_1, v_2, ..., v_n. 
    Returns a list of vectors, whose length is no greater than the space dimension. 
    Note that the vectors are stores "row-wise" in these lists.
    '''
    dim = vectors[0].size
    ret = [np.array(v) for v in vectors]
    final_vectors = []
    for i in range(len(vectors)):
        for j in range(i):
            ret[i] -= ret[i].dot(ret[j]) * ret[j]
        if np.linalg.norm(ret[i]) > 1e-10:
            ret[i] = normalize(ret[i])
            final_vectors.append(ret[i])
    return final_vectors

# ------------------------------------------------------------------------------------------------
# (Diophantine) equations over integers, number theory, etc.
# ------------------------------------------------------------------------------------------------

def gcd_list(numbers: List[int]) -> int:
    "Greatest common divisor of a set of integers"
    if len(numbers) == 0:
        return 1
    elif len(numbers) == 1:
        return numbers[0]
    ret = numbers[0]
    for n in numbers[1:]:
        ret = gcd(ret, n)
    return ret

def lcm_list(numbers: List[int]) -> int:
    "Least common multiple of a set of integers"
    if len(numbers) == 0:
        return 0
    gcd = gcd_list(numbers)
    if gcd == 0:  # This happens only if all the numbers are zero
        return 0
    prod = 1
    for n in numbers:
        prod *= n
    return prod / gcd ** (len(numbers) - 1)

def reduce_to_coprimes(numbers: Sequence[int]) -> List[int]:
    '''Reduce a greatest common divisor d in a set of integers numbers = {n1, n2, ..., nk} 
    s.t. {n1/d, n2/d, ..., nk/d} are coprime integers, i.e., gcd(n1/d,...,nk/d) = 1'''
    d = gcd_list(numbers)
    return list(n_i // d for n_i in numbers)

def solve_LDE_2vars_homo(a: int, b: int) -> tuple:
    '''Solve a linear Diophantine equation ax + by = 0 over integers x, y.
    Returns a fundamental solution (x0, y0) s.t. the general one has the form (n x0, n y0), 
    with an arbitrary integer n. Assumes that (a, b) != (0, 0).
    '''
    if a == 0 and b == 0:  # Not handling this degenerate case
        raise ValueError('solve_LDE_2vars_homo(): A Diophantine equation should have at least one nonzero coefficient')
    return tuple(reduce_to_coprimes([b, -a]))

def generalized_euclidean_algorithm(a: int, b: int) -> Tuple[int]:
    "Find a particular solution (x, y) of ax + by = 1, assuming that gcd(a,b) = 1"
    if a == 0 and b == 0:
        raise ValueError('generalized_euclidean_algorithm(): at least one of the coefficients should be nonzero')
    if gcd(a, b) != 1:
        return None # No solutions
    # Reduce to positive a >= b
    aa, bb = abs(a), abs(b)
    if aa < bb:
        aa, bb = bb, aa
        should_swap = True
    else:
        should_swap = False
    if bb == 0:    # Then inevitably aa = 1
        x, y = 1, 0
    elif bb == 1:  # aa = bb = 1
        x, y = 0, 1
    else:
        # Euclidean algorithm: if a % b = r and a // b = q, then r = a - qb & do the same for b,r instead of a,b
        r_n = aa % bb
        r_nminus1 = bb
        x_n, y_n = 1, -(aa // bb)      # r_n = aa x_n + bb y_n 
        x_nminus1, y_nminus1 = 0, 1    # i.e., bb = r_nminus1 = aa * 0 + bb * 1
        while r_n > 1:
            q_nplus1, r_nplus1 = divmod(r_nminus1, r_n)
            x_nminus1, y_nminus1, x_n, y_n = x_n, y_n, x_nminus1 - q_nplus1 * x_n, y_nminus1 - q_nplus1 * y_n
            r_nminus1, r_n = r_n, r_nplus1
        if r_n != 1:   # For safety
            raise ValueError(f'generalized_euclidean_algorithm(): internal error, final remainder {r_n} != 1')
        x, y = x_n, y_n 
    # Return x, y, correcting them for the possibly negative a, b
    if should_swap:
        x, y = y, x
    return (x if a >= 0 else -x), (y if b >= 0 else -y)
    

def solve_LDE_2vars_inhomo(a: int, b: int, c: int) -> Union[tuple, None]:
    '''Solve a linear Diophantine equation ax + by = c over integers x, y.
    Returns a tuple ((x0, y0), (u0, v0)), where (x0, y0) is a patricular solution and the general one 
    has the form (x0 + n u0, y0 + n v0), with an arbitrary integer n. Assumes that (a, b) != (0, 0).
    In case there are no solution, None is returned.
    '''
    if a == 0 and b == 0:  # Not handling this degenerate case
        raise ValueError('solve_LDE_2vars_homo(): A Diophantine equation should have at least one nonzero coefficient')
    elif c == 0:  # A homogeneous equation
        return ((0, 0), solve_LDE_2vars_homo(a, b))
    # Step 1: check if the solutions exist and, if possible, reduce the coefficients
    div = gcd(a, b)
    if c % gcd(a, b) != 0:
        return None
    aa, bb, cc = a // div, b // div, c // div
    # Step 2: apply the generalized Euclidean algorithm to find the particular solution (x0, y0)
    # Namely, we solve aa x0 + bb y0 = 1 (gcd(aa, bb) = 1) and multiply the result by cc.
    x0, y0 = generalized_euclidean_algorithm(aa, bb)
    x0 *= cc
    y0 *= cc
    # Step 3: find a fundamental solution (u0, v0) from a homogeneous LDE ax + by = 0 and return
    return ((x0, y0), solve_LDE_2vars_homo(aa, bb))


def solve_LDE_3vars_homo(a: int, b: int, c: int) -> tuple:
    '''Solve a linear Diophantine equation ax + by + cz = 0 over integers x, y, z.
    Returns a tuple ((u0, v0, w0), (u1, v1, w1)), where s.t. the general solution reads 
    (m u0 + n u1, m v0 + n v1, m w0 + n w1) for arbitrary integers m, n. 
    Assumes that (a, b, c) != (0, 0, 0). In case there are no solution, None is returned.
    '''
    if a == 0 and b == 0 and c == 0:  # Not handling this degenerate case
        raise ValueError('solve_LDE_3vars_homo(): A Diophantine equation should have at least one nonzero coefficient')
    elif a == 0 and b == 0:           # A special case cz = 0 handled separately
        return ((1, 0, 0), (0, 1, 0))
    # Now factor out as much as we can and reduce to an inhomogeneours bivariate LDE aa x + bb y = -cc z
    aa, bb, cc = reduce_to_coprimes([a, b, c])
    div = gcd(aa, bb)    # Note that gcd(div, cc) = 1, so div should be contained in z = div * xi
    (x2D, y2D), (u2D, v2D) = solve_LDE_2vars_inhomo(aa // div, bb // div, -cc)  # A solution for xi = 1
    return ((x2D, y2D, div), (u2D, v2D, 0))


# ------------------------------------------------------------------------------------------------
# Rational approximations of real numbers
# ------------------------------------------------------------------------------------------------


def snap_to_rational(x: float, deltax: float = 0.01) -> Fraction:
    "Replace a real number x by a rational one m/n s.t. |x - m/n| <= deltax and the denominator m is as small as possible"
    if deltax >= 0.5:
        return Fraction(round(x))
    # Note that in [x - deltax, x + deltax], there is always at least one rational with n >= 0.5 / deltax
    # and there are no more than one rational with n < 0.5 / deltax
    # A dummy version 0.1: just a brute-force approach
    n_max = floor(0.5 / deltax)
    for n in range(1, n_max + 1):
        x_rounded = round(x * n) / n
        if abs(x_rounded - x) <= deltax:
            return Fraction(round(x * n), n)
    # For safety
    raise ValueError('In snap_to_rational(): internal error, could not find a rational approximation')

