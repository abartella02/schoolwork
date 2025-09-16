import sympy as sp
from sympy import sin, cos, pi


def transf_matrix(theta='theta', d='d', a='a', alpha='alpha'):
    _theta, _d, _a, _alpha = sp.symbols('theta, d, a, alpha')

    m = sp.Matrix([
        [cos(_theta), -sin(_theta)*cos(_alpha), sin(_theta)*sin(_alpha), _a*cos(_theta)],
        [sin(_theta), cos(_theta)*cos(_alpha), -cos(_theta)*sin(_alpha), _a*sin(_theta)],
        [0         , sin(_alpha)           , cos(_alpha)            , _d           ],
        [0         , 0                    , 0                     , 1           ]
    ])

    args = [theta, d, a, alpha]
    symbols = [_theta, _d, _a, _alpha]
    subs = [(symbols[i], args[i]) for i, _ in enumerate(args)]
    return m.subs(subs)


def part1():
    zto = f"\N{SUPERSCRIPT ZERO}T\N{SUBSCRIPT ONE}"
    ott = f"\N{SUPERSCRIPT ONE}T\N{SUBSCRIPT TWO}"
    ttt = f"\N{SUPERSCRIPT TWO}T\N{SUBSCRIPT THREE}"

    zero_T_one = transf_matrix(pi/2, 'a1', 0, pi/2)
    one_T_two = transf_matrix('theta1', 0, 0, pi/2)
    two_T_three = transf_matrix(0, 'a2', 0, 0)

    print(zto)
    sp.pprint(zero_T_one)
    print()
    print(ott)
    sp.pprint(one_T_two)
    print()
    print(ttt)
    sp.pprint(two_T_three)
    print()

    print("Operations")
    print("----------")
    print(f"{zto} * {ott}")
    sp.pprint(res := zero_T_one*one_T_two)
    print()
    print(f"({zto} * {ott}) * {ttt}")
    sp.pprint(res * two_T_three)


def part2():
    zto = f"\N{SUPERSCRIPT ZERO}T\N{SUBSCRIPT ONE}"
    ott = f"\N{SUPERSCRIPT ONE}T\N{SUBSCRIPT TWO}"
    ttt = f"\N{SUPERSCRIPT TWO}T\N{SUBSCRIPT THREE}"

    zero_T_one = transf_matrix('theta0', 0, 'a1', 0)
    one_T_two = transf_matrix('theta1', 0, 'a2', 0)
    two_T_three = transf_matrix('theta2', 0, 'a3', 0)

    print(zto)
    sp.pprint(zero_T_one)
    print()
    print(ott)
    sp.pprint(one_T_two)
    print()
    print(ttt)
    sp.pprint(two_T_three)
    print()

    print("Operations")
    print("----------")
    print(f"{zto} * {ott}")
    sp.pprint(res := zero_T_one*one_T_two)
    print()
    print(f"({zto} * {ott}) * {ttt}")
    sp.pprint(sp.trigsimp(res * two_T_three))


if __name__ == '__main__':
    part1()


