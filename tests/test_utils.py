# =================================================================================================
# Test_Utils.py: unit tests for utils.py module
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

import numpy as np
from math import sqrt, pi

from src.utils import *
from src.utils import _all_units
from src.units import *

separ = '-' * 100

if __name__ == '__main__':
    print('==========================================================================================\n' +
          '                             Testing utils.py module...\n' +
          '==========================================================================================')
    print('Testing homogeneous 2-variable Diophantine solver solve_LDE_2vars_homo(),\n' +
          'which finds a fundamental solution (x, y) of ax + by = 0 over integers...')
    current_test = failed_tests = succeeded_tests = 0
    for a, b in np.random.randint(-25, 25, size=(50, 2)):
        current_test += 1
        if a == 0 and b == 0:
            print(f'\tTest #{current_test}: a = b = 0, skipping a degenerate LDE...')
            continue
        sol = solve_LDE_2vars_homo(a, b)
        subst = (a * sol[0] + b * sol[1] == 0)
        if subst == True:
            succeeded_tests += 1
        else:
            failed_tests += 1
        print(f'\tTest #{current_test}: a = {a}, b = {b}: solve_LDE_2vars_homo(a, b) = {sol}; ' +
              f'solution substitution correct: {subst}')
    print(f'{succeeded_tests}/{succeeded_tests + failed_tests} tests succeeded')
    
    print('------------------------------------------------------------------------------------------')
    print('Testing generalized Euclidean algorithm generalized_euclidean_algorithm(), namely,\n' +
          'a Diophantine solver ax + by = 1...')
    current_test = failed_tests = succeeded_tests = 0
    for a, b in np.random.randint(-100, 100, size=(50, 2)):
        current_test += 1
        if a == 0 and b == 0 or gcd(a, b) != 1:
            print(f'\tTest #{current_test}: a = b = 0 or a and b aren\'t coprime, skipping test...')
            continue
        print(f'\tTest #{current_test}: a = {a}, b = {b}: ', end='')
        x, y = generalized_euclidean_algorithm(a, b)
        subst = (a * x + b * y == 1)
        if subst == True:
            succeeded_tests += 1
        else:
            failed_tests += 1
        print(f'generalized_euclidean_algorithm(a, b) = {(x, y)}; solution substitution correct: {subst}')
    print(f'{succeeded_tests}/{succeeded_tests + failed_tests} tests succeeded')
    
    print('------------------------------------------------------------------------------------------')
    print('Testing inhomogeneous 2-variable Diophantine solver solve_LDE_2vars_inhomo(),\n' +
          'which finds a fundamental solution (x, y) of ax + by = c over integers...')
    current_test = failed_tests = succeeded_tests = 0
    for a, b, c in np.random.randint(-100, 100, size=(50, 3)):
        current_test += 1
        if a == 0 and b == 0:
            print(f'\tTest #{current_test}: a = b = 0, skipping test...')
            continue
        print(f'\tTest #{current_test}: a = {a}, b = {b}, c = {c}: ', end='')
        solution = solve_LDE_2vars_inhomo(a, b, c)
        if solution == None:
            print(f'solve_LDE_2vars_inhomo() = None, hard to check, skipping...')
            continue
        (x, y), (u, v) = solution
        print(f'part_sol = {(x, y)}, sol_shift = {(u, v)}', end='')
        subst = True 
        for n in np.random.randint(-100, 100, size=10):
            subst = subst and (a * (x + n * u) + b * (y + n * v) == c)
            if subst == False:
                break
        if subst == True:
            succeeded_tests += 1
        else:
            failed_tests += 1
        print(f'; solution substitution correct: {subst}')
    print(f'{succeeded_tests}/{succeeded_tests + failed_tests} tests succeeded')

    print('------------------------------------------------------------------------------------------')
    print('Testing homogeneous 3-variable Diophantine solver solve_LDE_3vars_homo(),\n' +
          'which finds two fundamental solutions (x, y, z) of ax + by + cz = 0 over integers...')
    current_test = failed_tests = succeeded_tests = 0
    for a, b, c in np.random.randint(-100, 100, size=(50, 3)):
        current_test += 1
        if a == 0 and b == 0 and c == 0:
            print(f'\tTest #{current_test}: a = b = c = 0, skipping test...')
            continue
        print(f'\tTest #{current_test}: a = {a}, b = {b}, c = {c}: ', end='')
        solution1, solution2 = solve_LDE_3vars_homo(a, b, c)
        print(f'sol_1 = {solution1}, sol_2 = {solution2}', end='')
        subst = True 
        for m, n in np.random.randint(-100, 100, size=(10, 2)):
            subst = subst and np.dot([a, b, c], m * np.array(solution1) + n * np.array(solution2)) == 0
            if subst == False:
                break
        if subst == True:
            succeeded_tests += 1
        else:
            failed_tests += 1
        print(f'; solution substitution correct: {subst}')
    print(f'{succeeded_tests}/{succeeded_tests + failed_tests} tests succeeded')

    print('------------------------------------------------------------------------------------------')
    print('Testing rational approximations snap_to_rational(x) -> m/n...')
    current_test = failed_tests = succeeded_tests = 0
    # Each test is of the form [x, deltax, m, n]
    golden_mean = 0.5 * (1 + sqrt(5.0))
    tests = [[pi, 0.01, 22, 7], [pi, 0.03, 19, 6], [pi, 0.1, 16, 5], [pi, 0.001, 201, 64], 
             [1234.0, 0.43, 1234, 1], [12.0, 5, 12, 1], [0.0, 0.0001, 0, 1],
             [golden_mean, 0.01, 13, 8], [golden_mean, 0.005, 21, 13], [golden_mean, 0.001, 55, 34]
            ]
    for x, deltax, m, n in tests:
        current_test += 1
        print(f'\tTest #{current_test}: x = {x}, deltax = {deltax}: ', end='')
        sol = snap_to_rational(x, deltax)
        subst = (sol == Fraction(m, n))
        print(f'snap_to_rational(x, deltax) = {sol}; solution correct = ', subst)
        if subst:
            succeeded_tests += 1
        else:
            failed_tests += 1
    print(f'{succeeded_tests}/{succeeded_tests + failed_tests} tests succeeded')

    print('------------------------------------------------------------------------------------------')
    print('Testing dimensionful quantity parser...')
    current_test = failed_tests = succeeded_tests = 0
    # Each test is of the form [string, value|None, default_units, allowed_units], None meaning a syntax error
    tests = [['1.0', 1.0, None, None], ['1.0^2', None, None, None], ['-3.5e10 erg', -3.5e10 * erg, None, None],
             ['9.81 m/sec^2', 9.81 * meter / sec**2, None, {'m', 'cm', 'sec', 'min'}],
             ['9.81 cm/hr^2', None, None, {'m', 'cm', 'sec', 'min'}],
             ['1.05e-34 J * sec', 1.05e-27 * erg * sec, None, {'J', 'sec'}],
             ['1.05e-34 J*sec', 1.05e-27 * erg * sec, None, {'J', 'sec'}],
             ['1.05e-34 J sec', None, None, {'J', 'sec'}],
             ['1.05e-34', 1.05e-27 * erg * sec, Joule * sec, {'J', 'sec'}],
             ['1.05e-34', 1.05e-27 * erg * sec, Joule * sec, {'J', 'sec'}]
            ]
    for s, val, def_units, allowed_units in tests:
        current_test += 1
        if def_units is None:
            def_units = 1.0
        if allowed_units is None:
            allowed_units = _all_units.keys()
        print(f'\tTest #{current_test}: s = {repr(s)}, default_units = {def_units}, allowed_units = {allowed_units}: ', end='')
        try:
            sol = float_with_units(s, default_units=def_units, allowed_units=allowed_units)
            print(f', result = {sol: .3g}', end='')
            error = False
        except:
            print(f', result = [syntax error]', end='')
            error = True
        if (val is None and error) or (val != None and not error and isinstance(sol, float) and abs(sol / val - 1.0) < 1e-12):
            succeeded_tests += 1
            print(', correct = True')
        else:
            failed_tests += 1
            print(', correct = False')
    print(f'{succeeded_tests}/{succeeded_tests + failed_tests} tests succeeded')
        
