"""Description of all gadget kinds and their underlying circuit."""

import abc
from typing import Sequence
import itertools as it
from typing import Type

import numpy as np
import scipy.stats

import gates

INCLUDE_COPY_GATES = True


def gadget_n_probe_distr(n: int, p: float):
    binom = scipy.stats.binom(n, p)
    return binom.pmf(np.arange(0, n + 1))


def gadget_n_probe_distr_group_t(n: int, p: float, tsuff: int):
    """Like n_probe_distr but group all cases with >= tsuff probes."""
    res = gadget_n_probe_distr(n, p)
    if n > tsuff:
        res = np.concatenate((res[:tsuff], [np.sum(res[tsuff:])]))
    return res


class Gadget:
    _all_names = set()

    def __init__(self, name: str):
        assert name not in self._all_names, f"gadget name {name} already used"
        self._all_names.add(name)
        self._name = name

    def __repr__(self):
        # return f"Gadget(name={self.name()})"
        return self.name()

    def set_output_uses(self, uses: list["Gadget"]):
        self._n_output_uses = 0
        for g in uses:
            if not isinstance(g, OutputSharing):
                assert not isinstance(g, InputSharing)
                self._n_output_uses += 1

    @abc.abstractmethod
    def output_sharing(self) -> list[gates.Gate]:
        pass

    @abc.abstractmethod
    def n_shares(self) -> int:
        pass

    @abc.abstractmethod
    def extended_probes(self, gate_leakage: bool) -> list[list[gates.Gate]]:
        """Extended probes in the gadget

        If gate_leakage is True, then there is one extended probe per gate and
        the extended probe contains all the operands of the gate.
        Otherwise, each extended probe contains a single wire. There is one
        extended probe for each operand of a gate. Further, there are extended
        probes for copy gates.
        If a gate in the gadget outputs a wire that is used n times (combining
        usage inside and outside the gadget), we assume that the gadget
        contains n-1 additional copy gates for that wire, which each have one
        extended probe on that wire.
        """
        pass

    def n_probes(self, gate_leakage: bool) -> int:
        return len(self.extended_probes(gate_leakage))

    def n_probe_distr(self, p: float, gate_leakage: bool):
        return gadget_n_probe_distr(self.n_probes(gate_leakage), p)

    def n_probe_distr_group_t(self, p: float, tsuff: int, gate_leakage: bool):
        """Like n_probe_distr but group all cases with >= tsuff probes."""
        return gadget_n_probe_distr_group_t(self.n_probes(gate_leakage), p, tsuff)

    def leaky(self, gate_leakage: bool) -> bool:
        # False for input sharings.
        return self.n_probes(gate_leakage) != 0

    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def inputs(self) -> "Sequence[Gadget]":
        pass


class NoSimGadget(Gadget):
    """Gadget without simulation, has no computation inside."""

    pass


class SimGadget(Gadget):
    @abc.abstractmethod
    def t_sni(self) -> int:
        pass


# We need input sharings for and initial circuit variables and for the wire
# leakage model (where the input gates are included in the set of gates).
# We do not need output sharings: there is no leakage associated to these sharings.
class InputSharing(SimGadget):
    def __init__(self, name: str, nshares: int):
        super().__init__(name)
        self.shares = gates.xor_sharing(name, nshares)
        self.d = nshares

    def n_shares(self):
        return self.d

    def extended_probes(self, gate_leakage: bool) -> list[list[gates.Gate]]:
        if gate_leakage or not INCLUDE_COPY_GATES:
            return []
        else:
            return max(0, self._n_output_uses - 1) * [[s] for s in self.shares]

    def output_sharing(self) -> list[gates.Gate]:
        return self.shares

    def inputs(self) -> "Sequence[Gadget]":
        return []

    def t_sni(self) -> int:
        # Even though we're not SNI, we don't care about how inputs propagate
        # to the inputs since this gadget represents the inputs themselves.
        # The important property is that we can simulate without knowing the
        # unmasked input if there are at most d-1 probes.
        return self.d - 1


class OutputSharing(NoSimGadget):
    def __init__(self, name: str, input: Gadget):
        super().__init__(name)
        self.input = input
        self.d = input.n_shares()
        self.shares = input.output_sharing()

    def n_shares(self):
        return self.d

    def extended_probes(self, gate_leakage: bool) -> list[list[gates.Gate]]:
        return []

    def output_sharing(self) -> list[gates.Gate]:
        raise NotImplementedError()

    def inputs(self) -> "Sequence[Gadget]":
        return [self.input]


class OpGadget(SimGadget):
    def __init__(self, name: str, *operands: Gadget):
        super().__init__(name)
        assert len(operands) == self.arity()
        op_sharings = [op.output_sharing() for op in operands]
        self.d = len(op_sharings[0])
        assert all(len(x) == self.d for x in op_sharings)
        self.operands = operands
        self.output_shares, self.l_gates = self.compute_circuit(op_sharings)
        assert all(isinstance(sh, gates.Gate) for sh in self.output_shares)

    @classmethod
    @abc.abstractmethod
    def arity(cls) -> int:
        pass

    @abc.abstractmethod
    def compute_circuit(
        self, op_sharings: Sequence[Sequence[gates.Gate]]
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        """
        op_sharings: for each operand, the list of its shares
        returns (output_shares, internal gates)
        """
        pass

    @classmethod
    def sanity_check(cls, n_shares: int):
        inputs = [
            InputSharing(f"test_{repr(cls)}{n_shares}_{i}", n_shares)
            for i in range(cls.arity())
        ]
        gadget = cls("test" + repr(cls), *inputs)
        assert len(gadget.output_sharing()) == n_shares
        probes_ref = set(
            tuple(g.leak_gate(gate_leakage=True))
            for g in set().union(*(o.set_of_probes() for o in gadget.output_sharing()))
        )
        probes_ref = {p for p in probes_ref if p}
        probes_test = set(map(tuple, gadget.extended_probes(gate_leakage=True)))
        assert probes_test == probes_ref, (probes_test, probes_ref)

    def n_shares(self):
        return self.d

    def output_sharing(self) -> list[gates.Gate]:
        return self.output_shares

    def inputs(self) -> "Sequence[Gadget]":
        return self.operands

    def extended_probes(self, gate_leakage: bool) -> list[list[gates.Gate]]:
        if gate_leakage or not INCLUDE_COPY_GATES:
            return [g.leak_gate(gate_leakage) for g in self.l_gates]
        else:
            # We need to account for leakage from copy gates.
            # The output of a gate used n times is copied n-1 times,
            # hence we need to add the leakage of n-1 copy gates (the leakage
            # from the uses themselves is already accounted for elsewhere).
            gate_uses = {g: 0 for g in self.l_gates}
            for g in self.l_gates:
                for op in g._operands():
                    if op in gate_uses:
                        gate_uses[op] += 1
            for s in self.output_shares:
                gate_uses[s] += self._n_output_uses
            copy_gate_leakage = [
                [g] for g, nu in gate_uses.items() for _ in range(nu - 1)
            ]
            return copy_gate_leakage + [g.leak_gate(gate_leakage) for g in self.l_gates]

    def t_sni(self) -> int:
        return self.d - 1


def make_leaking_gate(kind: gates.OpKind, gate_set: list[gates.Gate]):
    def make_gate(prefix: str, ops: list[gates.Gate]):
        res = gates.OpGate.new_unique(prefix, kind, ops)
        gate_set.append(res)
        return res

    return make_gate


def isw_compress(
    products: list[list[gates.Gate]],
) -> tuple[list[gates.Gate], list[gates.Gate]]:
    d = len(products)
    r = [
        [gates.RndGate.new_unique(f"r{i}{j}") for j in range(i + 1, d)]
        for i in range(d)
    ]
    c = []
    circ = []
    xor_g = make_leaking_gate(gates.OpKind.ADD, circ)
    for i in range(d):
        acc = products[i][i]
        for j in range(d):
            if i != j:
                aibj = products[i][j]
                r_aibj = xor_g(f"xr{i}{j}", [aibj, r[min(i, j)][abs(i - j) - 1]])
                acc = xor_g(f"acc{i}{j}", [acc, r_aibj])
        c.append(acc)
    return c, circ


class Refresh(OpGadget):
    @classmethod
    def arity(cls) -> int:
        return 1

    @classmethod
    @abc.abstractmethod
    def circuit(
        cls,
        x: Sequence[gates.Gate],
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        pass

    def compute_circuit(self, op_sharings):
        (x,) = op_sharings
        return self.circuit(x)


class ZeroSharingRefresh(Refresh):
    @classmethod
    def circuit(
        cls,
        x: Sequence[gates.Gate],
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        if len(x) == 1:
            return list(x), []
        r, lr = cls._zero_sharing(len(x))
        assert len(r) == len(x)
        y = [
            gates.OpGate.new_unique(f"refxor{i}", gates.OpKind.ADD, [xi, ri])
            for i, (xi, ri) in enumerate(zip(x, r))
        ]
        return y, lr + y

    @classmethod
    @abc.abstractmethod
    def _zero_sharing(cls, n: int) -> tuple[list[gates.Gate], list[gates.Gate]]:
        pass


class HalfRefresh(Refresh):
    @classmethod
    def circuit(
        cls,
        x: Sequence[gates.Gate],
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        if len(x) == 1:
            return list(x), []
        n = len(x) // 2
        r = [gates.RndGate.new_unique(f"ref{n}_{i}") for i in range(n)]
        y = [
            gates.OpGate.new_unique(f"refxor{i}", gates.OpKind.ADD, [xi, ri])
            for i, (xi, ri) in enumerate(zip(x, r + r))
        ]
        z = y if len(y) == len(x) else y + [x[-1]]
        assert len(z) == len(x)
        return z, y


class HalfRefresh2(Refresh):
    @classmethod
    def circuit(
        cls,
        x: Sequence[gates.Gate],
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        if len(x) == 1:
            return list(x), []
        n = len(x) // 2
        r = [gates.RndGate.new_unique(f"ref{n}_{i}") for i in range(n)]
        y = [
            gates.OpGate.new_unique(f"refxor{i}", gates.OpKind.ADD, [xi, ri])
            for i, (xi, ri) in enumerate(zip(x, r + r))
        ]
        if len(y) == len(x):
            z = y
        else:
            r = gates.RndGate.new_unique(f"ref{n}_{n}")
            z0 = gates.OpGate.new_unique("refxor_21", gates.OpKind.ADD, [y[0], r])
            z_last = gates.OpGate.new_unique("refxor_22", gates.OpKind.ADD, [x[-1], r])
            z = [z0] + y[1:] + [z_last]
            y = y + [z0, z_last]
        assert len(z) == len(x)
        return z, y


class SimpleRefresh(ZeroSharingRefresh):
    @classmethod
    def _zero_sharing(cls, n: int) -> tuple[list[gates.Gate], list[gates.Gate]]:
        assert n >= 2
        r = [gates.RndGate.new_unique(f"ref{n}_{i}") for i in range(n - 1)]
        r_sum = r[0]
        leaks = []
        for i, ri in enumerate(r[1:]):
            r_sum = gates.OpGate.new_unique(
                f"ref{n}s{i}x", gates.OpKind.ADD, [r_sum, ri]
            )
            leaks.append(r_sum)
        return r + [r_sum], leaks


class CircRefresh(ZeroSharingRefresh):
    @classmethod
    def _zero_sharing(cls, n: int) -> tuple[list[gates.Gate], list[gates.Gate]]:
        if n <= 2:
            return SimpleRefresh._zero_sharing(n)
        else:
            r = [gates.RndGate.new_unique(f"ref{n}_{i}") for i in range(n)]
            res = [
                gates.OpGate.new_unique(
                    f"refc{n}s{i}x", gates.OpKind.ADD, [r[i], r[(i + 1) % n]]
                )
                for i in range(n)
            ]
            return res, res


class NlognRefresh(ZeroSharingRefresh):
    @classmethod
    def _zero_sharing(cls, n: int) -> tuple[list[gates.Gate], list[gates.Gate]]:
        assert n >= 2
        if n == 2:
            r = gates.RndGate.new_unique("ref2")
            return [r, r], []
        elif n == 3:
            r0 = gates.RndGate.new_unique("ref3_0")
            r1 = gates.RndGate.new_unique("ref3_1")
            s = gates.OpGate.new_unique("ref3x", gates.OpKind.ADD, [r0, r1])
            return [r1, s, r0], [s]
        else:
            r0, l0 = cls._zero_sharing(n // 2)
            r1, l1 = cls._zero_sharing(n - (n // 2))
            r = [gates.RndGate.new_unique(f"ref{n}_{i}") for i in range(n // 2)]
            s0 = [
                gates.OpGate.new_unique(f"ref{n}p0{i}x", gates.OpKind.ADD, [r0i, ri])
                for i, (r0i, ri) in enumerate(zip(r0, r))
            ]
            s1 = [
                gates.OpGate.new_unique(f"ref{n}p1{i}x", gates.OpKind.ADD, [r1i, ri])
                for i, (r1i, ri) in enumerate(zip(r1, r))
            ]
            if len(s1) < len(r1):
                s1.append(r1[-1])
            s = s0 + s1
            return s, l0 + l1 + s


class IswMult(OpGadget):
    @classmethod
    def arity(cls) -> int:
        return 2

    def compute_circuit(self, op_sharings):
        return self.circuit(op_sharings)

    @staticmethod
    def circuit(
        op_sharings: Sequence[Sequence[gates.Gate]],
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        a, b = op_sharings
        d = len(a)
        circ = []
        and_g = make_leaking_gate(gates.OpKind.MUL, circ)
        products = [
            [and_g(f"c{i}{j}", [a[i], b[j]]) for j in range(d)] for i in range(d)
        ]
        c, circ_compress = isw_compress(products)
        return c, circ + circ_compress


class IswAndNot(OpGadget):
    @classmethod
    def arity(cls) -> int:
        return 2

    def compute_circuit(self, op_sharings):
        return self.circuit(op_sharings)

    @staticmethod
    def circuit(
        op_sharings: Sequence[Sequence[gates.Gate]],
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        a, b = op_sharings
        notb = [gates.OpGate.new_unique("not", gates.OpKind.NEG, [b[0]])]
        notb += list(b)[1:]
        c, lc = IswMult.circuit([a, notb])
        return c, [notb[0]] + lc


MatRef = list[list[tuple[gates.Gate, gates.Gate]]]


def matref(
    a: list[gates.Gate], b: list[gates.Gate], refresh
) -> tuple[MatRef, list[gates.Gate]]:
    assert len(a) != 0 and len(b) != 0
    if len(a) == 1 and len(b) == 1:
        return [[(a[0], b[0])]], []
    a_s = [a[: len(a) // 2], a[len(a) // 2 :]]
    b_s = [b[: len(b) // 2], b[len(b) // 2 :]]
    c = []
    m = {0: dict(), 1: dict()}
    for i, j in it.product(range(2), range(2)):
        if not a_s[i]:
            m[i][j] = []
        elif not b_s[j]:
            m[i][j] = [[] for _ in a_s[i]]
        else:
            ai, ai_c = refresh(a_s[i])
            bj, bj_c = refresh(b_s[j])
            m[i][j], c_m = matref(ai, bj, refresh)
            c += ai_c + bj_c + c_m
    res = [m0 + m1 for m0, m1 in zip(m[0][0] + m[1][0], m[0][1] + m[1][1])]
    return res, c

def matref_2_refresh(
    a: list[gates.Gate], b: list[gates.Gate], refresh1, refresh2
) -> tuple[MatRef, list[gates.Gate]]:
    assert len(a) != 0 and len(b) != 0
    if len(a) == 1 and len(b) == 1:
        return [[(a[0], b[0])]], []
    a_s = [a[: len(a) // 2], a[len(a) // 2 :]]
    b_s = [b[: len(b) // 2], b[len(b) // 2 :]]
    c = []
    m = {0: dict(), 1: dict()}
    for i, j in it.product(range(2), range(2)):
        if not a_s[i]:
            m[i][j] = []
        elif not b_s[j]:
            m[i][j] = [[] for _ in a_s[i]]
        else:
            ai, ai_c = refresh1(a_s[i])
            bj, bj_c = refresh2(b_s[j])
            m[i][j], c_m = matref(ai, bj, refresh1, refresh2)
            c += ai_c + bj_c + c_m
    res = [m0 + m1 for m0, m1 in zip(m[0][0] + m[1][0], m[0][1] + m[1][1])]
    return res, c

class FullMatRefMult(OpGadget):
    def __init__(self, name: str, *operands: Gadget, refresh: Type[Refresh]):
        self._refresh = refresh
        super().__init__(name, *operands)

    @classmethod
    def arity(cls) -> int:
        return 2

    def compute_circuit(self, op_sharings):
        return self.circuit(op_sharings, self._refresh)

    @staticmethod
    def circuit(
        op_sharings: Sequence[Sequence[gates.Gate]],
        refresh,
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        a, b = op_sharings
        m, circ = matref(list(a), list(b), refresh.circuit)
        d = len(a)
        and_g = make_leaking_gate(gates.OpKind.MUL, circ)
        products = [
            [and_g(f"c{i}{j}", list(m[i][j])) for j in range(d)] for i in range(d)
        ]
        c, circ_compress = isw_compress(products)
        return c, circ + circ_compress


class LinearGadget(OpGadget):
    @abc.abstractmethod
    def linear_op(
        cls, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        pass

    def compute_circuit(
        self, op_sharings: Sequence[Sequence[gates.Gate]]
    ) -> tuple[list[gates.Gate], list[gates.Gate]]:
        c, list_leaks = zip(
            *(self.linear_op(i, shares) for i, shares in enumerate(zip(*op_sharings)))
        )
        d, lref = NlognRefresh.circuit(c)
        return d, [leak for leaks in list_leaks for leak in leaks] + lref


class Xor(LinearGadget):
    @classmethod
    def arity(cls) -> int:
        return 2

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        x = gates.OpGate.new_unique(f"xor{share_idx}", gates.OpKind.ADD, shares)
        return (x, [x])


class MulCst(LinearGadget):
    def __init__(self, constant: gates.CstGate, *args, **kwargs):
        self._constant = constant
        super().__init__(*args, **kwargs)

    @classmethod
    def arity(cls) -> int:
        return 1

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        x = gates.OpGate.new_unique(
            f"mul_cst_m{share_idx}", gates.OpKind.MUL, [shares[0], self._constant]
        )
        return (x, [x])
