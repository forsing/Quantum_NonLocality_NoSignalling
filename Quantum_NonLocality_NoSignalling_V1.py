from __future__ import annotations

# Quantum_NonLocality_NoSignalling

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit
import scipy

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("/data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBERS = 39
PICKS = 7

RIDGES = (0.1, 1.0)

MAX_ITERATIONS = 100
MAX_FUNCTION_CALLS = 600
CHECKPOINT_EVERY = 5
GRADIENT_LIMIT = 5.0e-5
ANGLE_BOUND = math.pi

CHANNEL_ERROR_DIAGNOSTICS = (
    0.0,
    0.1,
    0.3,
    0.49,
    0.5,
)

VERSION = "QUANTUM_NONLOCALITY_NOSIGNALLING_V1_20261029"

CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "Quantum_NonLocality_NoSignalling_V1_checkpoints"
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, message):
    require(np.all(np.isfinite(values)), message)


def binary_entropy(probability):
    probability = float(probability)

    require(
        0.0 <= probability <= 1.0,
        "Argument binarne entropije je izvan intervala.",
    )

    if probability == 0.0 or probability == 1.0:
        return 0.0

    return (
        -probability * math.log2(probability)
        - (1.0 - probability)
        * math.log2(1.0 - probability)
    )


def mutual_information(joint):
    joint = np.asarray(joint, dtype=float)

    require(
        np.all(np.isfinite(joint))
        and np.all(joint >= 0.0)
        and abs(float(np.sum(joint)) - 1.0) < 1.0e-10,
        "Neispravna zajednička raspodela za informacionu meru.",
    )

    first = np.sum(joint, axis=1)
    second = np.sum(joint, axis=0)
    independent = np.outer(first, second)

    positive = joint > 0.0

    require(
        np.all(independent[positive] > 0.0),
        "Neispravne marginale informacione mere.",
    )

    result = float(
        np.sum(
            joint[positive]
            * np.log2(
                joint[positive]
                / independent[positive]
            )
        )
    )

    require(
        result >= -1.0e-10,
        "Negativna uzajamna informacija.",
    )

    return max(0.0, result)


class FixedSizeDistribution:
    """
    Kvantni protokol daje q_i i r_i.

    Za legalan skup S:
        P(S) proporcionalna je proizvodu q_i za i u S
        i proizvodu r_i za i van S.

    Uslovljavanje na tačno k brojeva daje:
        log w_i = log q_i - log r_i.

    Normalizacija obuhvata sve legalne podskupove.
    """

    def __init__(self, numbers, picks):
        self.m = int(numbers)
        self.k = int(picks)

        require(
            0 <= self.k <= self.m,
            "Neispravna veličina skupa.",
        )

    def partition(self, log_weights, marginals=False):
        log_weights = np.asarray(
            log_weights,
            dtype=float,
        )

        require(
            log_weights.ndim == 2
            and log_weights.shape[1] == self.m
            and len(log_weights) > 0,
            "Neispravan oblik logaritamskih težina.",
        )

        finite(
            log_weights,
            "Logaritamske težine nisu konačne.",
        )

        count = len(log_weights)

        forward = np.full(
            (count, self.m + 1, self.k + 1),
            -np.inf,
        )

        forward[:, 0, 0] = 0.0

        for number in range(self.m):
            forward[:, number + 1] = forward[:, number]

            if self.k > 0:
                forward[:, number + 1, 1:] = np.logaddexp(
                    forward[:, number, 1:],
                    log_weights[:, number, None]
                    + forward[:, number, :self.k],
                )

        log_z = forward[:, self.m, self.k]

        finite(
            log_z,
            "Normalizaciona konstanta nije konačna.",
        )

        if not marginals:
            return log_z, None

        if self.k == 0:
            return log_z, np.zeros_like(log_weights)

        backward = np.full_like(forward, -np.inf)
        backward[:, self.m, 0] = 0.0

        for number in range(self.m - 1, -1, -1):
            backward[:, number] = backward[:, number + 1]

            backward[:, number, 1:] = np.logaddexp(
                backward[:, number + 1, 1:],
                log_weights[:, number, None]
                + backward[:, number + 1, :self.k],
            )

        require(
            np.max(
                np.abs(
                    backward[:, 0, self.k] - log_z
                )
            ) < 2.0e-10,
            "Dve normalizacione rekurzije se ne slažu.",
        )

        inclusion = np.empty_like(log_weights)

        for number in range(self.m):
            rest = logsumexp(
                forward[:, number, :self.k]
                + backward[
                    :,
                    number + 1,
                    self.k - 1::-1,
                ],
                axis=1,
            )

            inclusion[:, number] = np.exp(
                log_weights[:, number]
                + rest
                - log_z
            )

        finite(
            inclusion,
            "Marginalne verovatnoće nisu konačne.",
        )

        require(
            np.all(inclusion >= 0.0)
            and np.all(inclusion <= 1.0 + 2.0e-8),
            "Marginalna verovatnoća je izvan granica.",
        )

        require(
            np.max(
                np.abs(
                    np.sum(inclusion, axis=1) - self.k
                )
            ) < 2.0e-7,
            "Zbir marginala nije jednak broju izbora.",
        )

        return log_z, inclusion

    def log_probabilities(self, log_weights, targets):
        require(
            targets.shape == log_weights.shape
            and np.all(
                (targets == 0.0)
                | (targets == 1.0)
            )
            and np.all(
                np.sum(targets, axis=1) == self.k
            ),
            "Neispravni ciljni skupovi.",
        )

        log_z, _ = self.partition(log_weights)

        result = (
            np.sum(
                targets * log_weights,
                axis=1,
            )
            - log_z
        )

        finite(
            result,
            "Logaritamske verovatnoće nisu konačne.",
        )

        require(
            np.all(result <= 2.0e-8),
            "Verovatnoća skupa prelazi jedan.",
        )

        return result

    def maximum(self, log_weights):
        log_weights = np.asarray(
            log_weights,
            dtype=float,
        )

        require(
            log_weights.shape == (self.m,),
            "Neispravan oblik MAP težina.",
        )

        finite(log_weights, "MAP težine nisu konačne.")

        order = np.argsort(
            -log_weights,
            kind="stable",
        )

        selected = tuple(
            int(value)
            for value in np.sort(order[:self.k])
        )

        log_z, _ = self.partition(
            log_weights[None, :]
        )

        log_probability = (
            float(np.sum(log_weights[list(selected)]))
            - float(log_z[0])
        )

        require(
            -math.log(math.comb(self.m, self.k)) - 2.0e-8
            <= log_probability
            <= 2.0e-8,
            "MAP verovatnoća ne prolazi granice.",
        )

        return selected, log_probability

    def alternate_log_z(self, log_weights):
        log_weights = np.asarray(
            log_weights,
            dtype=np.longdouble,
        )

        shift = np.max(log_weights)
        weights = np.exp(log_weights - shift)

        coefficients = np.zeros(
            self.k + 1,
            dtype=np.longdouble,
        )

        coefficients[0] = 1.0

        for number, weight in enumerate(weights):
            for degree in range(
                min(self.k, number + 1),
                0,
                -1,
            ):
                coefficients[degree] += (
                    weight * coefficients[degree - 1]
                )

        require(
            coefficients[self.k] > 0.0,
            "Alternativna normalizacija nije pozitivna.",
        )

        return float(
            np.log(coefficients[self.k])
            + self.k * shift
        )


DISTRIBUTION = FixedSizeDistribution(NUMBERS, PICKS)


def cross_information(past, future):
    """
    Matrica I(past_i; future_j), samo za trening prefiks.

    Koristi se za izbor parova ulaznih bitova.
    Ne koristi se kao izlazna prognoza.
    """

    count = len(past)

    require(
        count > 0
        and past.shape == future.shape,
        "Neispravan skup za izbor parova.",
    )

    px = np.mean(past, axis=0)[:, None]
    py = np.mean(future, axis=0)[None, :]

    p11 = (past.T @ future) / count
    p10 = px - p11
    p01 = py - p11
    p00 = 1.0 - px - py + p11

    result = np.zeros_like(p11)

    entries = (
        (p11, px * py),
        (p10, px * (1.0 - py)),
        (p01, (1.0 - px) * py),
        (p00, (1.0 - px) * (1.0 - py)),
    )

    for joint, independent in entries:
        require(
            np.min(joint) >= -1.0e-12,
            "Neispravna verovatnoća pri izboru parova.",
        )

        positive = joint > 0.0

        require(
            np.all(independent[positive] > 0.0),
            "Neispravna marginala pri izboru parova.",
        )

        result[positive] += (
            joint[positive]
            * np.log2(
                joint[positive]
                / independent[positive]
            )
        )

    finite(
        result,
        "Informacioni skor parova nije konačan.",
    )

    return result


def training_blocks(past, future):
    scores = cross_information(past, future)

    candidates = []

    for first in range(NUMBERS):
        for second in range(first + 1, NUMBERS):
            score = (
                scores[first, second]
                + scores[second, first]
            )

            candidates.append(
                (-float(score), first, second)
            )

    candidates.sort()

    used = set()
    blocks = []

    for _, first, second in candidates:
        if first not in used and second not in used:
            blocks.append((first, second))
            used.add(first)
            used.add(second)

    for number in range(NUMBERS):
        if number not in used:
            blocks.append((number,))

    blocks.sort()

    covered = [
        number
        for block in blocks
        for number in block
    ]

    require(
        sorted(covered) == list(range(NUMBERS)),
        "Parovi ne pokrivaju sve brojeve tačno jednom.",
    )

    return tuple(blocks)


def direction(polar, azimuth):
    st = math.sin(polar)
    ct = math.cos(polar)
    sp = math.sin(azimuth)
    cp = math.cos(azimuth)

    vector = np.asarray(
        (st * cp, st * sp, ct)
    )

    derivatives = np.asarray(
        (
            (ct * cp, ct * sp, -st),
            (-st * sp, st * cp, 0.0),
        )
    )

    return vector, derivatives


class QuantumRAC:
    """
    q = 1: običan Bellov par.

    Alice:
        dobija x0, x1;
        bira lokalnu mernu bazu;
        dobija a;
        šalje m = x0 XOR a.

    Bob:
        dobija upit y;
        bira svoju lokalnu bazu;
        dobija b;
        vraća g = m XOR b.

    Aliceina merna baza ne zavisi od Bobovog upita.
    Svaki Bobov upit koristi zasebnu pripremu para.
    """

    def __init__(self, blocks, numbers=NUMBERS):
        self.blocks = tuple(
            tuple(block)
            for block in blocks
        )

        self.m = int(numbers)
        self.groups = len(self.blocks)
        self.alice_size = self.groups * 4 * 2
        self.size = self.alice_size + self.groups * 2 * 2

        covered = [
            number
            for block in self.blocks
            for number in block
        ]

        require(
            sorted(covered) == list(range(self.m))
            and all(
                len(block) in (1, 2)
                for block in self.blocks
            ),
            "Neispravna struktura kvantnih protokola.",
        )

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        self.bell = bell

    def initial_parameters(self):
        return np.zeros(self.size)

    def unpack(self, parameters):
        parameters = np.asarray(
            parameters,
            dtype=float,
        )

        require(
            parameters.shape == (self.size,),
            "Pogrešan broj mernih parametara.",
        )

        finite(
            parameters,
            "Merni parametri nisu konačni.",
        )

        alice = parameters[:self.alice_size].reshape(
            self.groups,
            4,
            2,
        )

        bob = parameters[self.alice_size:].reshape(
            self.groups,
            2,
            2,
        )

        return alice, bob

    def encode(self, past, older):
        require(
            past.shape == older.shape
            and past.ndim == 2
            and past.shape[1] == self.m,
            "Neispravan istorijski ulaz protokola.",
        )

        codes = np.empty(
            (len(past), self.groups),
            dtype=np.int64,
        )

        for group, block in enumerate(self.blocks):
            first = block[0]

            x0 = past[:, first].astype(np.int64)

            if len(block) == 2:
                x1 = past[:, block[1]].astype(np.int64)
            else:
                x1 = older[:, first].astype(np.int64)

            codes[:, group] = x0 + 2 * x1

        require(
            np.all(codes >= 0)
            and np.all(codes <= 3),
            "Neispravan dvobitni kod.",
        )

        return codes

    def lookup(self, parameters):
        alice_offsets, bob_offsets = self.unpack(parameters)

        q = np.empty((self.groups, 4, 2))
        r = np.empty_like(q)

        jacobian_alice = np.empty(
            (self.groups, 4, 2, 2)
        )

        jacobian_bob = np.empty_like(jacobian_alice)

        reflection = np.asarray((1.0, -1.0, 1.0))

        maximum_probability_error = 0.0
        maximum_no_signalling_error = 0.0

        for group in range(self.groups):
            bob_vectors = []
            bob_derivatives = []
            bob_readouts = []

            for query in range(2):
                polar = (
                    math.pi / 4.0
                    + query * math.pi / 2.0
                    + bob_offsets[group, query, 0]
                )

                azimuth = bob_offsets[group, query, 1]

                vector, derivative = direction(
                    polar, azimuth
                )

                bob_vectors.append(vector)
                bob_derivatives.append(derivative)

                readout = QuantumCircuit(2)
                readout.rz(-float(azimuth), 1)
                readout.ry(-float(polar), 1)

                bob_readouts.append(readout)

            for code in range(4):
                x0 = code & 1
                sign = 1.0 - 2.0 * x0

                polar = (
                    math.pi / 2.0
                    + alice_offsets[group, code, 0]
                )

                azimuth = (
                    math.pi / 2.0
                    + alice_offsets[group, code, 1]
                )

                alice_vector, alice_derivative = direction(
                    polar, azimuth
                )

                circuit = self.bell.copy()
                circuit.rz(-float(azimuth), 0)
                circuit.ry(-float(polar), 0)

                alice_state = Statevector.from_instruction(circuit)

                for query in range(2):
                    state = alice_state.evolve(
                        bob_readouts[query]
                    )

                    probabilities = np.abs(state.data) ** 2

                    require(
                        np.all(np.isfinite(probabilities))
                        and abs(
                            float(np.sum(probabilities)) - 1.0
                        ) < 1.0e-10,
                        "Qiskit zajednička raspodela nije normalizovana.",
                    )

                    guess = np.asarray(
                        [
                            x0
                            ^ (index & 1)
                            ^ ((index >> 1) & 1)
                            for index in range(4)
                        ]
                    )

                    q[group, code, query] = float(
                        np.sum(
                            probabilities[guess == 1]
                        )
                    )

                    r[group, code, query] = float(
                        np.sum(
                            probabilities[guess == 0]
                        )
                    )

                    bob_one = float(
                        probabilities[2] + probabilities[3]
                    )

                    maximum_no_signalling_error = max(
                        maximum_no_signalling_error,
                        abs(bob_one - 0.5),
                    )

                    bob_vector = bob_vectors[query]
                    bob_derivative = bob_derivatives[query]

                    correlation = float(
                        alice_vector
                        @ (reflection * bob_vector)
                    )

                    analytic_q = (
                        0.5 - 0.5 * sign * correlation
                    )

                    maximum_probability_error = max(
                        maximum_probability_error,
                        abs(
                            analytic_q
                            - q[group, code, query]
                        ),
                    )

                    jacobian_alice[group, code, query] = (
                        -0.5
                        * sign
                        * (
                            alice_derivative
                            @ (reflection * bob_vector)
                        )
                    )

                    jacobian_bob[group, code, query] = (
                        -0.5
                        * sign
                        * (
                            bob_derivative
                            @ (reflection * alice_vector)
                        )
                    )

        require(
            maximum_probability_error < 2.0e-10,
            "Analitički izraz odstupa od Qiskit komunikacionog protokola.",
        )

        require(
            maximum_no_signalling_error < 2.0e-10,
            "Bobove marginale pre poruke zavise od Aliceinog izbora.",
        )

        require(
            np.all(q > 0.0)
            and np.all(r > 0.0)
            and np.max(np.abs(q + r - 1.0)) < 1.0e-10,
            "Kvantni protokol je na nevažećoj granici verovatnoće.",
        )

        errors = {
            "qiskit": maximum_probability_error,
            "no_signalling": maximum_no_signalling_error,
        }

        return (
            q,
            r,
            jacobian_alice,
            jacobian_bob,
            errors,
        )

    def select(self, q, r, codes):
        require(
            codes.shape[1] == self.groups,
            "Neispravan broj protokola pri dekodiranju.",
        )

        selected_q = np.empty(
            (len(codes), self.m)
        )

        selected_r = np.empty_like(selected_q)

        for group, block in enumerate(self.blocks):
            for query, number in enumerate(block):
                selected_q[:, number] = q[
                    group,
                    codes[:, group],
                    query,
                ]

                selected_r[:, number] = r[
                    group,
                    codes[:, group],
                    query,
                ]

        return selected_q, selected_r

    def parameter_gradient(
        self,
        codes,
        derivative_q,
        jacobian_alice,
        jacobian_bob,
    ):
        table_gradient = np.zeros(
            (self.groups, 4, 2)
        )

        for group, block in enumerate(self.blocks):
            for query, number in enumerate(block):
                np.add.at(
                    table_gradient[group, :, query],
                    codes[:, group],
                    derivative_q[:, number],
                )

        alice_gradient = np.sum(
            table_gradient[:, :, :, None]
            * jacobian_alice,
            axis=2,
        )

        bob_gradient = np.sum(
            table_gradient[:, :, :, None]
            * jacobian_bob,
            axis=1,
        )

        return np.concatenate(
            (
                alice_gradient.ravel(),
                bob_gradient.ravel(),
            )
        )


def protocol_information(q_table, channel_error):
    """
    Zaseban kalibracioni eksperiment.

    Aliceini bitovi su ovde nezavisni i ravnomerno zadati.
    To nije pretpostavka o CSV podacima.

    Greške kanala prosečavaju se analitički.
    Ne izvlače se slučajne greške.
    """

    require(
        q_table.shape == (4, 2)
        and 0.0 <= channel_error <= 0.5,
        "Neispravan kalibracioni kanal.",
    )

    information = 0.0
    success = 0.0

    for query in range(2):
        joint = np.zeros((2, 2))

        for code in range(4):
            true_bit = (code >> query) & 1

            p_one = (
                channel_error
                + (1.0 - 2.0 * channel_error)
                * q_table[code, query]
            )

            joint[true_bit, 0] += 0.25 * (1.0 - p_one)
            joint[true_bit, 1] += 0.25 * p_one

        information += mutual_information(joint)

        success += (
            0.5
            * float(joint[0, 0] + joint[1, 1])
        )

    capacity = 1.0 - binary_entropy(channel_error)

    return information, capacity, success


def check_information_causality(q):
    maximum_excess = -math.inf

    for group in range(len(q)):
        for channel_error in CHANNEL_ERROR_DIAGNOSTICS:
            information, capacity, _ = protocol_information(
                q[group],
                channel_error,
            )

            excess = information - capacity
            maximum_excess = max(maximum_excess, excess)

            require(
                excess <= 2.0e-9,
                "Kalibracioni protokol krši informacionu kauzalnost.",
            )

    return max(0.0, maximum_excess)


def objective(
    parameters,
    source,
    codes,
    targets,
    distribution,
    ridge,
):
    q, r, ja, jb, _ = source.lookup(parameters)

    selected_q, selected_r = source.select(
        q, r, codes
    )

    log_weights = (
        np.log(selected_q)
        - np.log(selected_r)
    )

    log_z, inclusion = distribution.partition(
        log_weights,
        marginals=True,
    )

    log_probability = (
        np.sum(targets * log_weights, axis=1)
        - log_z
    )

    value = -float(np.mean(log_probability))

    derivative_q = (
        (inclusion - targets)
        * (
            1.0 / selected_q
            + 1.0 / selected_r
        )
        / len(targets)
    )

    gradient = source.parameter_gradient(
        codes,
        derivative_q,
        ja,
        jb,
    )

    value += (
        0.5
        * ridge
        * float(np.mean(parameters ** 2))
    )

    gradient += ridge * parameters / len(parameters)

    require(
        np.isfinite(value),
        "Funkcija cilja nije konačna.",
    )

    finite(
        gradient,
        "Gradijent mernih uglova nije konačan.",
    )

    return value, gradient


def checkpoint_key(source, codes, targets, ridge):
    digest = hashlib.sha256()

    configuration = (
        VERSION,
        qiskit.__version__,
        np.__version__,
        scipy.__version__,
        source.blocks,
        source.m,
        MAX_ITERATIONS,
        MAX_FUNCTION_CALLS,
        ANGLE_BOUND,
        float(ridge),
    )

    digest.update(
        repr(configuration).encode("utf-8")
    )

    for values in (codes, targets):
        array = np.ascontiguousarray(values)

        digest.update(
            repr(
                (array.shape, str(array.dtype))
            ).encode("utf-8")
        )

        digest.update(array.tobytes())

    return digest.hexdigest()


def save_checkpoint(
    path,
    parameters,
    iterations,
    done,
    message,
):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray(VERSION),
            parameters=parameters,
            iterations=np.asarray(iterations),
            done=np.asarray(done),
            message=np.asarray(message),
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def projected_gradient_norm(parameters, gradient):
    result = gradient.copy()

    result[
        (parameters <= -ANGLE_BOUND + 1.0e-10)
        & (result > 0.0)
    ] = 0.0

    result[
        (parameters >= ANGLE_BOUND - 1.0e-10)
        & (result < 0.0)
    ] = 0.0

    return float(np.max(np.abs(result)))


@dataclass
class Fitted:
    source: QuantumRAC
    parameters: np.ndarray
    q: np.ndarray
    r: np.ndarray
    loss: float
    gradient_norm: float
    converged: bool
    message: str


def fit_model(
    past,
    older,
    targets,
    ridge,
    label,
):
    blocks = training_blocks(past, targets)
    source = QuantumRAC(blocks)
    codes = source.encode(past, older)

    path = (
        CHECKPOINT_DIRECTORY
        / f"{checkpoint_key(source, codes, targets, ridge)}.npz"
    )

    initial = source.initial_parameters()
    parameters = initial.copy()

    used = 0
    done = False
    message = "Početni kandidat"

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    saved["version"].item() == VERSION,
                    "Nepodudarna verzija kontrolne tačke.",
                )

                candidate = saved["parameters"].copy()
                candidate_used = int(saved["iterations"].item())

                require(
                    candidate.shape == initial.shape
                    and np.all(np.isfinite(candidate))
                    and np.all(
                        np.abs(candidate)
                        <= ANGLE_BOUND + 1.0e-10
                    )
                    and 0 <= candidate_used <= MAX_ITERATIONS,
                    "Neispravna kontrolna tačka.",
                )

                candidate_done = bool(saved["done"].item())
                candidate_message = str(saved["message"].item())

            objective(
                candidate,
                source,
                codes,
                targets,
                DISTRIBUTION,
                ridge,
            )

            parameters = candidate
            used = candidate_used
            done = candidate_done
            message = candidate_message

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            BadZipFile,
            EOFError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            say(
                f"Kontrolna tačka nije upotrebljiva: {label}",
                str(error),
            )

            parameters = initial.copy()
            used = 0
            done = False
            message = "Početak bez neispravne kontrolne tačke"

    value, gradient = objective(
        parameters,
        source,
        codes,
        targets,
        DISTRIBUTION,
        ridge,
    )

    best = {
        "parameters": parameters.copy(),
        "value": float(value),
        "gradient": gradient.copy(),
    }

    prior_iterations = used
    local_iterations = 0
    started = time.monotonic()

    say(
        "",
        label,
        f"Trening prelaza: {len(targets)}",
        f"Kvantnih protokola: {source.groups}",
        f"Mernih parametara: {source.size}",
        f"Regularizacija: {ridge}",
    )

    if done:
        say("Koristim završenu kontrolnu tačku.")
    else:
        save_checkpoint(
            path,
            best["parameters"],
            used,
            False,
            message,
        )

    def fun(candidate):
        candidate_value, candidate_gradient = objective(
            candidate,
            source,
            codes,
            targets,
            DISTRIBUTION,
            ridge,
        )

        if candidate_value < best["value"]:
            best["parameters"] = np.asarray(
                candidate,
                dtype=float,
            ).copy()

            best["value"] = float(candidate_value)
            best["gradient"] = candidate_gradient.copy()

        return candidate_value, candidate_gradient

    def callback(_candidate):
        nonlocal local_iterations, used

        local_iterations += 1

        used = min(
            MAX_ITERATIONS,
            prior_iterations + local_iterations,
        )

        if local_iterations % CHECKPOINT_EVERY == 0:
            norm = projected_gradient_norm(
                best["parameters"],
                best["gradient"],
            )

            say(
                f"Iteracija {used}/{MAX_ITERATIONS}"
                f" | cilj {best['value']:.9f}"
                f" | Max|proj.grad| {norm:.3e}"
                f" | proteklo {time.monotonic() - started:.1f} s"
            )

            save_checkpoint(
                path,
                best["parameters"],
                used,
                False,
                "Trening je u toku",
            )

    remaining = MAX_ITERATIONS - used

    if not done and remaining > 0:
        try:
            result = minimize(
                fun,
                best["parameters"],
                method="L-BFGS-B",
                jac=True,
                bounds=[
                    (-ANGLE_BOUND, ANGLE_BOUND)
                ] * source.size,
                callback=callback,
                options={
                    "maxiter": remaining,
                    "maxfun": MAX_FUNCTION_CALLS,
                    "maxls": 35,
                    "maxcor": 15,
                    "ftol": 1.0e-12,
                    "gtol": 1.0e-6,
                },
            )

            used = min(
                MAX_ITERATIONS,
                prior_iterations + int(result.nit),
            )

            message = str(result.message)

        except KeyboardInterrupt:
            save_checkpoint(
                path,
                best["parameters"],
                used,
                False,
                "Korisnički prekid",
            )
            raise

        except (
            RuntimeError,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            message = (
                "Zadržan najbolji validan kandidat. "
                f"Razlog prekida optimizacije: {error}"
            )

    elif not done:
        message = "Iskorišćen je budžet iteracija."

    value, gradient = objective(
        best["parameters"],
        source,
        codes,
        targets,
        DISTRIBUTION,
        ridge,
    )

    norm = projected_gradient_norm(
        best["parameters"],
        gradient,
    )

    q, r, _, _, _ = source.lookup(
        best["parameters"]
    )

    save_checkpoint(
        path,
        best["parameters"],
        used,
        True,
        message,
    )

    say(
        f"Zadržani cilj: {value:.9f}",
        f"Max|proj.grad|: {norm:.6e}",
        f"Status optimizatora: {message}",
    )

    if norm > GRADIENT_LIMIT:
        say(
            "DIJAGNOSTIKA: gradijentni uslov nije potvrđen.",
            "Kandidat ima validnu raspodelu; minimum nije potvrđen.",
        )

    return Fitted(
        source=source,
        parameters=best["parameters"].copy(),
        q=q.copy(),
        r=r.copy(),
        loss=float(value),
        gradient_norm=norm,
        converged=norm <= GRADIENT_LIMIT,
        message=message,
    )


def model_weights(model, past, older):
    codes = model.source.encode(past, older)

    q, r = model.source.select(
        model.q,
        model.r,
        codes,
    )

    return np.log(q) - np.log(r)


def evaluate(model, past, older, targets, hits=False):
    log_weights = model_weights(
        model,
        past,
        older,
    )

    log_probability = DISTRIBUTION.log_probabilities(
        log_weights,
        targets,
    )

    nll = -float(np.mean(log_probability))
    histogram = None

    if hits:
        selected = np.argsort(
            -log_weights,
            axis=1,
            kind="stable",
        )[:, :PICKS]

        matches = np.sum(
            np.take_along_axis(
                targets,
                selected,
                axis=1,
            ),
            axis=1,
        ).astype(np.int64)

        histogram = np.bincount(
            matches,
            minlength=PICKS + 1,
        )

        require(
            int(np.sum(histogram)) == len(targets),
            "Neispravna statistika pogodaka.",
        )

    return nll, histogram


def load_draws(path):
    fingerprint = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    rows = []
    empty = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if not row or all(
                not item.strip()
                for item in row
            ):
                empty += 1
                continue

            require(
                len(row) == PICKS,
                f"{path.name}, red {line_number}: očekujem sedam kolona.",
            )

            try:
                values = [
                    int(item.strip())
                    for item in row
                ]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(values)) == PICKS
                and all(
                    1 <= number <= NUMBERS
                    for number in values
                ),
                f"{path.name}, red {line_number}: nelegalna sedmorka.",
            )

            rows.append(
                sorted(number - 1 for number in values)
            )

    require(
        len(rows) >= 100,
        f"{path.name}: premalo redova.",
    )

    draws = np.asarray(
        rows,
        dtype=np.int64,
    )

    occupancy = np.zeros(
        (len(draws), NUMBERS)
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {empty}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled CSV-a.",
    )

    return draws, occupancy


def self_checks():
    say(
        "",
        "Pokrećem ugrađene provere pre treninga.",
    )

    distribution = FixedSizeDistribution(6, 3)

    weights = np.asarray(
        [[0.07, 0.13, 0.19, 0.23, 0.17, 0.21]]
    )

    log_weights = np.log(weights)

    log_z, inclusion = distribution.partition(
        log_weights,
        marginals=True,
    )

    legal = list(combinations(range(6), 3))

    raw = np.asarray(
        [
            np.prod(weights[0, list(subset)])
            for subset in legal
        ]
    )

    probabilities = raw / np.sum(raw)

    require(
        abs(
            log_z[0]
            - math.log(float(np.sum(raw)))
        ) < 1.0e-11,
        "Normalizacija odstupa od male enumeracije.",
    )

    expected_inclusion = np.zeros(6)

    for subset, probability in zip(legal, probabilities):
        expected_inclusion[list(subset)] += probability

    require(
        np.max(
            np.abs(
                inclusion[0] - expected_inclusion
            )
        ) < 1.0e-11,
        "Marginale odstupaju od enumeracije.",
    )

    selected, log_probability = distribution.maximum(
        log_weights[0]
    )

    best_index = int(np.argmax(probabilities))

    require(
        selected == legal[best_index]
        and abs(
            log_probability
            - math.log(probabilities[best_index])
        ) < 1.0e-11,
        "Maksimum odstupa od enumeracije.",
    )

    for picks in (0, 1, 6):
        boundary = FixedSizeDistribution(6, picks)

        _, marginal = boundary.partition(
            log_weights,
            marginals=True,
        )

        require(
            abs(
                float(np.sum(marginal)) - picks
            ) < 1.0e-10,
            "Neispravan granični slučaj.",
        )

    calibration = QuantumRAC(
        ((0, 1),),
        numbers=2,
    )

    ideal = calibration.initial_parameters()
    alice, _ = calibration.unpack(ideal)

    for code in range(4):
        xor = (code & 1) ^ ((code >> 1) & 1)

        desired_polar = (
            0.0 if xor else math.pi / 2.0
        )

        alice[0, code, 0] = (
            desired_polar - math.pi / 2.0
        )

        alice[0, code, 1] = -math.pi / 2.0

    ideal_q, _, _, _, ideal_errors = calibration.lookup(
        ideal
    )

    expected_success = (
        1.0 + 1.0 / math.sqrt(2.0)
    ) / 2.0

    information, capacity, success = protocol_information(
        ideal_q[0],
        0.0,
    )

    require(
        abs(success - expected_success) < 2.0e-10,
        "Idealna dvobitna rekonstrukcija nije usaglašena.",
    )

    require(
        information <= capacity + 2.0e-9,
        "Idealni protokol krši informacionu kauzalnost.",
    )

    check_information_causality(ideal_q)

    zero_q, _, _, _, _ = calibration.lookup(
        calibration.initial_parameters()
    )

    require(
        np.max(np.abs(zero_q - 0.5)) < 2.0e-10,
        "Početni protokol nije neutralan.",
    )

    no_information, no_capacity, no_success = protocol_information(
        ideal_q[0],
        0.5,
    )

    require(
        abs(no_information) < 2.0e-10
        and abs(no_capacity) < 2.0e-10
        and abs(no_success - 0.5) < 2.0e-10,
        "Granični slučaj kanala bez kapaciteta nije ispravan.",
    )

    source = QuantumRAC(
        ((0, 3), (1, 4), (2,)),
        numbers=5,
    )

    past = np.asarray(
        (
            (1, 0, 0, 1, 0),
            (0, 1, 0, 0, 1),
            (1, 0, 1, 0, 0),
            (0, 0, 1, 1, 0),
        ),
        dtype=float,
    )

    older = np.vstack(
        (past[:1], past[:-1])
    )

    targets = np.asarray(
        (
            (0, 1, 0, 0, 1),
            (1, 0, 1, 0, 0),
            (0, 0, 1, 1, 0),
            (1, 0, 0, 0, 1),
        ),
        dtype=float,
    )

    codes = source.encode(past, older)
    test_distribution = FixedSizeDistribution(5, 2)

    parameters = (
        0.03
        * np.sin(
            np.arange(source.size) + 0.3
        )
    )

    _, analytic = objective(
        parameters,
        source,
        codes,
        targets,
        test_distribution,
        0.1,
    )

    epsilon = 1.0e-6
    maximum_gradient_error = 0.0

    for index in range(source.size):
        delta = np.zeros(source.size)
        delta[index] = epsilon

        plus, _ = objective(
            parameters + delta,
            source,
            codes,
            targets,
            test_distribution,
            0.1,
        )

        minus, _ = objective(
            parameters - delta,
            source,
            codes,
            targets,
            test_distribution,
            0.1,
        )

        numerical = (
            plus - minus
        ) / (2.0 * epsilon)

        error = abs(
            numerical - analytic[index]
        )

        maximum_gradient_error = max(
            maximum_gradient_error,
            error,
        )

        require(
            error
            < 5.0e-5
            * (1.0 + abs(analytic[index])),
            "Gradijent komunikaционog modela ne prolazi proveru.",
        )

    learned_q, _, _, _, _ = source.lookup(parameters)
    check_information_causality(learned_q)

    neutral_weights = np.zeros((1, NUMBERS))

    neutral_log_z, _ = DISTRIBUTION.partition(
        neutral_weights
    )

    expected_log_z = math.log(
        math.comb(NUMBERS, PICKS)
    )

    require(
        abs(
            neutral_log_z[0] - expected_log_z
        ) < 2.0e-10,
        "Predstavljanje skupova ne prolazi neutralnu proveru.",
    )

    say(
        "Normalizacija, marginale, granice i MAP: provereni na maloj enumeraciji.",
        f"Kalibraciona rekonstrukcija poznatih bitova: {success:.12f}",
        f"Kalibraciona suma informacija: {information:.12f}",
        f"Kapacitet kalibracionog kanala: {capacity:.12f}",
        f"Greška prekomunikacionih Bobovih marginala: {ideal_errors['no_signalling']:.3e}",
        f"Najveća proverena greška gradijenta: {maximum_gradient_error:.3e}",
        "Informaciona kauzalnost: proverena na odvojenom standardnom eksperimentu.",
    )


def longest_run(numbers):
    longest = 0
    current = 0
    previous = None

    for number in numbers:
        if previous is not None and number == previous + 1:
            current += 1
        else:
            current = 1

        longest = max(longest, current)
        previous = number

    return longest


def process_file(path, draws, occupancy):
    previous = np.vstack(
        (occupancy[:1], occupancy[:-1])
    )

    past = occupancy[:-1]
    older = previous[:-1]
    targets = occupancy[1:]

    count = len(targets)

    s60 = int(0.60 * count)
    s70 = int(0.70 * count)
    s80 = int(0.80 * count)

    require(
        20 < s60 < s70 < s80 < count,
        "Neispravne hronološke podele.",
    )

    records = []

    for ridge in RIDGES:
        weighted_nll = 0.0
        evaluated = 0

        for fold, (train_end, validation_end) in enumerate(
            ((s60, s70), (s70, s80)),
            start=1,
        ):
            model = fit_model(
                past[:train_end],
                older[:train_end],
                targets[:train_end],
                ridge,
                (
                    f"{path.name} / validacija {fold}"
                    f" / ridge {ridge}"
                ),
            )

            nll, _ = evaluate(
                model,
                past[train_end:validation_end],
                older[train_end:validation_end],
                targets[train_end:validation_end],
            )

            fold_count = validation_end - train_end

            weighted_nll += nll * fold_count
            evaluated += fold_count

            say(
                f"Validacija {fold} NLL: {nll:.9f}"
            )

        mean_nll = weighted_nll / evaluated

        records.append(
            (float(mean_nll), float(ridge))
        )

        say(
            f"Objedinjeni validacioni NLL za ridge {ridge}: "
            f"{mean_nll:.9f}"
        )

    records.sort()
    validation_nll, chosen_ridge = records[0]

    say(
        "",
        f"{path.name}: izabrani ridge {chosen_ridge}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit_model(
        past[:s80],
        older[:s80],
        targets[:s80],
        chosen_ridge,
        f"{path.name} / model za odvojeni test",
    )

    test_nll, histogram = evaluate(
        test_model,
        past[s80:],
        older[s80:],
        targets[s80:],
        hits=True,
    )

    # Bez Aliceine poruke, Bobove kvantne marginale su 1/2.
    # Ova kontrola nije završni model niti pretpostavka o CSV-u.
    no_message_log_weights = np.zeros(
        (len(targets[s80:]), NUMBERS)
    )

    no_message_nll = -float(
        np.mean(
            DISTRIBUTION.log_probabilities(
                no_message_log_weights,
                targets[s80:],
            )
        )
    )

    final_model = fit_model(
        past,
        older,
        targets,
        chosen_ridge,
        f"{path.name} / završno učenje na svim prelazima",
    )

    q, r, _, _, errors = final_model.source.lookup(
        final_model.parameters
    )

    final_model.q = q.copy()
    final_model.r = r.copy()

    ic_excess = check_information_causality(q)

    next_log_weights = model_weights(
        final_model,
        occupancy[-1:],
        occupancy[-2:-1],
    )

    log_z, inclusion = DISTRIBUTION.partition(
        next_log_weights,
        marginals=True,
    )

    selected, next_log_probability = DISTRIBUTION.maximum(
        next_log_weights[0]
    )

    alternative_log_z = DISTRIBUTION.alternate_log_z(
        next_log_weights[0]
    )

    normalization_error = abs(
        alternative_log_z - float(log_z[0])
    )

    require(
        normalization_error < 2.0e-10,
        "Dve normalizacije NEXT raspodele se ne slažu.",
    )

    recent_indices = np.arange(
        max(0, len(occupancy) - 8),
        len(occupancy),
    )

    recent_log_weights = model_weights(
        final_model,
        occupancy[recent_indices],
        occupancy[
            np.maximum(recent_indices - 1, 0)
        ],
    )

    _, recent_inclusion = DISTRIBUTION.partition(
        recent_log_weights,
        marginals=True,
    )

    sensitivity = float(
        np.max(
            np.ptp(
                recent_inclusion,
                axis=0,
            )
        )
    )

    warnings = []

    if not test_model.converged:
        warnings.append(
            "Model za test nema potvrđen gradijentni uslov."
        )

    if not final_model.converged:
        warnings.append(
            "Završni model nema potvrđen gradijentni uslov."
        )

    test_improved = (
        test_nll < no_message_nll - 1.0e-8
    )

    if not test_improved:
        warnings.append(
            "Odvojeni test ne potvrđuje poboljšanje prema kontroli bez poruke."
        )

    run = longest_run(selected)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if selected[-1] - selected[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    selected_array = np.asarray(
        selected,
        dtype=np.int64,
    )

    if np.any(
        np.all(
            draws == selected_array[None, :],
            axis=1,
        )
    ):
        warnings.append(
            "NEXT se već pojavio u istoriji ovog CSV-a."
        )

    if np.array_equal(
        selected_array,
        draws[-1],
    ):
        warnings.append(
            "NEXT je jednak poslednjem kolu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Raspodela slabo reaguje na poslednje proverene istorijske ulaze."
        )

    mean_hits = float(
        np.dot(
            np.arange(PICKS + 1),
            histogram,
        ) / np.sum(histogram)
    )

    say(
        "",
        path.name,
        "NEXT: " + " ".join(
            str(number + 1)
            for number in selected
        ),
        f"Validacioni NLL: {validation_nll:.9f}",
        f"Odvojeni test NLL: {test_nll:.9f}",
        f"Test NLL kontrole bez poruke: {no_message_nll:.9f}",
        f"Modelna verovatnoća NEXT: {math.exp(next_log_probability):.12g}",
        f"Normalizacioni odnos: {math.exp(alternative_log_z - log_z[0]):.12f}",
        f"Greška dve normalizacije: {normalization_error:.3e}",
        f"Zbir marginalnih verovatnoća: {np.sum(inclusion[0]):.12f}",
        f"Greška Qiskit komunikacionog izraza: {errors['qiskit']:.3e}",
        f"Greška Bobovih marginala pre poruke: {errors['no_signalling']:.3e}",
        f"Najveće kalibraciono prekoračenje kapaciteta: {ic_excess:.3e}",
        f"Dijagnostička osetljivost: {sensitivity:.6e}",
        f"Završni Max|proj.grad|: {final_model.gradient_norm:.6e}",
        "Naučeni parovi ulaznih brojeva:",
        " | ".join(
            (
                f"{block[0] + 1}-{block[1] + 1}"
                if len(block) == 2
                else f"{block[0] + 1}: dva istorijska trenutka"
            )
            for block in final_model.source.blocks
        ),
        "Test: broj kola prema broju pogodaka",
        " | ".join(
            f"{hits}: {int(value)}"
            for hits, value in enumerate(histogram)
        ),
        f"Prosečan broj pogodaka MAP kombinacije: {mean_hits:.6f}",
        "Marginalne verovatnoće uključivanja brojeva:",
    )

    for start in range(0, NUMBERS, 6):
        say(
            " | ".join(
                f"{number + 1:02d}: {inclusion[0, number]:.8f}"
                for number in range(
                    start,
                    min(start + 6, NUMBERS),
                )
            )
        )

    say(
        "TEST STATUS: "
        + (
            "NIŽI NLL OD KONTROLE BEZ PORUKE"
            if test_improved
            else "POBOLJŠANJE NIJE POTVRĐENO"
        )
    )

    for warning in warnings:
        say(f"DIJAGNOSTIKA: {warning}")

    return {
        "name": path.name,
        "selected": selected,
        "inclusion": inclusion[0].copy(),
        "last": tuple(int(value) for value in draws[-1]),
        "sensitivity": sensitivity,
        "test_improved": test_improved,
    }


def main():
    say(
        "QUANTUM NON-LOCALITY VERSUS NO-SIGNALLING V1",
        "Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "",
        "Resurs q = 1: običan dvokubitni Bellov par.",
        "Alice dobija dva istorijska bita.",
        "Alice šalje jednu klasičnu poruku po protokolu.",
        "Bob prema upitu bira svoju mernu bazu.",
        "Dekodiranje koristi Aliceinu poruku i Bobov ishod.",
        "",
        "Parovi brojeva biraju se samo iz trening prefiksa.",
        "Izbor parova koristi informacioni skor, a ne izlazno rangiranje učestalosti.",
        "To je deterministički pohlepni izbor; optimalno uparivanje nije garantovano.",
        "Za jedan preostali broj koriste se dva istorijska trenutka.",
        "",
        "Svaki Bobov upit koristi zasebnu pripremu kvantnog para.",
        "Jedna poruka po paru nije jedan bit za svih 39 brojeva.",
        "Dva različita Bobova merenja ne pripisuju se istom primerku para.",
        "",
        "Uče se uglovi lokalnih kvantnih merenja.",
        "Izlazne verovatnoće računa Qiskit.",
        "Raspodela sedmorke je proizvod izlaznih Bernulijevih faktora",
        "uslovljen izborom tačno sedam različitih brojeva.",
        "Nema dodatnog naučenog sprezanja budućih brojeva između protokola.",
        "Normalizacija i MAP ove raspodele računaju se tačno.",
        "",
        "Predikcija koristi kanal bez šuma.",
        "Kapacitet kanala proverava se u odvojenom kalibracionom eksperimentu.",
        "Kalibracioni šum prosečava se analitički.",
        "Ne dodaje se šum CSV podacima ili predikciji.",
        "Kalibraciona pretpostavka nezavisnih bitova ne primenjuje se na loto podatke.",
        "",
        "Hronološka validacija bira regularizaciju.",
        "Odvojeni test ne bira parametre.",
        "Završno učenje koristi sve istorijske prelaze.",
        "Nema klasifikatora ili klasičnog regresora iza kvantnog izlaza.",
        "Optimizacija i računanje normalizacije su klasični.",
        "Optimizacija je lokalna; globalni minimum nije potvrđen.",
        "",
        "Nema slučajnog uzorkovanja merenja ili generisanja kombinacija.",
        "Nema kozmetičkih filtera za zbir, parnost, zone ili nizove.",
        "Numerička ispravnost i kvalitet testa ispisuju se zasebno.",
        "",
        f"Kontrolne tačke: {CHECKPOINT_DIRECTORY}",
    )

    loaded = []

    for path in FILES:
        draws, occupancy = load_draws(path)
        loaded.append((path, draws, occupancy))

    self_checks()

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for path, draws, occupancy in loaded:
        results.append(
            process_file(
                path,
                draws,
                occupancy,
            )
        )

    require(
        len(results) == len(FILES),
        "Nisu završena sva tri CSV-a.",
    )

    say("", "ZAVRŠNI NEXT REZULTATI")

    for result in results:
        say(
            result["name"]
            + " | NEXT: "
            + " ".join(
                str(number + 1)
                for number in result["selected"]
            )
            + " | "
            + (
                "NIŽI TEST NLL OD KONTROLE"
                if result["test_improved"]
                else "TEST POBOLJŠANJE NIJE POTVRĐENO"
            )
        )

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if (
                results[first]["selected"]
                == results[second]["selected"]
            ):
                difference = float(
                    np.max(
                        np.abs(
                            results[first]["inclusion"]
                            - results[second]["inclusion"]
                        )
                    )
                )

                same_last = (
                    results[first]["last"]
                    == results[second]["last"]
                )

                say(
                    "DIJAGNOSTIKA: isti NEXT za "
                    f"{results[first]['name']} i "
                    f"{results[second]['name']}.",
                    f"Najveća razlika NEXT marginala: {difference:.6e}",
                    f"Jednako poslednje istorijsko kolo: {same_last}",
                    "Osetljivosti na istoriju: "
                    f"{results[first]['sensitivity']:.6e} i "
                    f"{results[second]['sensitivity']:.6e}.",
                )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say(
            "",
            "PREKINUTO na zahtev korisnika.",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM NON-LOCALITY VERSUS NO-SIGNALLING V1
Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4

Resurs q = 1: običan dvokubitni Bellov par.
Alice dobija dva istorijska bita.
Alice šalje jednu klasičnu poruku po protokolu.
Bob prema upitu bira svoju mernu bazu.
Dekodiranje koristi Aliceinu poruku i Bobov ishod.

Parovi brojeva biraju se samo iz trening prefiksa.
Izbor parova koristi informacioni skor, a ne izlazno rangiranje učestalosti.
To je deterministički pohlepni izbor; optimalno uparivanje nije garantovano.
Za jedan preostali broj koriste se dva istorijska trenutka.

Svaki Bobov upit koristi zasebnu pripremu kvantnog para.
Jedna poruka po paru nije jedan bit za svih 39 brojeva.
Dva različita Bobova merenja ne pripisuju se istom primerku para.

Uče se uglovi lokalnih kvantnih merenja.
Izlazne verovatnoće računa Qiskit.
Raspodela sedmorke je proizvod izlaznih Bernulijevih faktora
uslovljen izborom tačno sedam različitih brojeva.
Nema dodatnog naučenog sprezanja budućih brojeva između protokola.
Normalizacija i MAP ove raspodele računaju se tačno.

Predikcija koristi kanal bez šuma.
Kapacitet kanala proverava se u odvojenom kalibracionom eksperimentu.
Kalibracioni šum prosečava se analitički.
Ne dodaje se šum CSV podacima ili predikciji.
Kalibraciona pretpostavka nezavisnih bitova ne primenjuje se na loto podatke.

Hronološka validacija bira regularizaciju.
Odvojeni test ne bira parametre.
Završno učenje koristi sve istorijske prelaze.
Nema klasifikatora ili klasičnog regresora iza kvantnog izlaza.
Optimizacija i računanje normalizacije su klasični.
Optimizacija je lokalna; globalni minimum nije potvrđen.

Nema slučajnog uzorkovanja merenja ili generisanja kombinacija.
Nema kozmetičkih filtera za zbir, parnost, zone ili nizove.
Numerička ispravnost i kvalitet testa ispisuju se zasebno.

Kontrolne tačke: /Quantum_NonLocality_NoSignalling/Quantum_NonLocality_NoSignalling_V1_checkpoints

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled CSV-a.

Pokrećem ugrađene provere pre treninga.
Normalizacija, marginale, granice i MAP: provereni na maloj enumeraciji.
Kalibraciona rekonstrukcija poznatih bitova: 0.853553390593
Kalibraciona suma informacija: 0.798247926614
Kapacitet kalibracionog kanala: 1.000000000000
Greška prekomunikacionih Bobovih marginala: 1.665e-16
Najveća proverena greška gradijenta: 3.028e-10
Informaciona kauzalnost: proverena na odvojenom standardnom eksperimentu.

loto7_4710_k86.csv / validacija 1 / ridge 0.1
Trening prelaza: 2825
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 0.1
Iteracija 5/100 | cilj 16.503302925 | Max|proj.grad| 1.832e-02 | proteklo 0.4 s
Iteracija 10/100 | cilj 16.499255877 | Max|proj.grad| 3.382e-03 | proteklo 0.8 s
Iteracija 15/100 | cilj 16.498948345 | Max|proj.grad| 9.940e-04 | proteklo 1.2 s
Iteracija 20/100 | cilj 16.498919284 | Max|proj.grad| 1.430e-04 | proteklo 1.6 s
Iteracija 25/100 | cilj 16.498914787 | Max|proj.grad| 5.718e-05 | proteklo 1.9 s
Iteracija 30/100 | cilj 16.498914181 | Max|proj.grad| 2.629e-05 | proteklo 2.3 s
Iteracija 35/100 | cilj 16.498913854 | Max|proj.grad| 6.965e-05 | proteklo 2.7 s
Iteracija 40/100 | cilj 16.498913189 | Max|proj.grad| 1.259e-04 | proteklo 3.0 s
Iteracija 45/100 | cilj 16.498912187 | Max|proj.grad| 4.766e-05 | proteklo 3.4 s
Iteracija 50/100 | cilj 16.498911884 | Max|proj.grad| 1.090e-05 | proteklo 3.8 s
Iteracija 55/100 | cilj 16.498911863 | Max|proj.grad| 1.775e-05 | proteklo 4.1 s
Iteracija 60/100 | cilj 16.498911859 | Max|proj.grad| 3.497e-06 | proteklo 4.5 s
Iteracija 65/100 | cilj 16.498911855 | Max|proj.grad| 4.173e-06 | proteklo 4.8 s
Iteracija 70/100 | cilj 16.498911843 | Max|proj.grad| 1.477e-05 | proteklo 5.2 s
Iteracija 75/100 | cilj 16.498911836 | Max|proj.grad| 4.202e-06 | proteklo 5.5 s
Iteracija 80/100 | cilj 16.498911835 | Max|proj.grad| 8.435e-07 | proteklo 5.9 s
Zadržani cilj: 16.498911835
Max|proj.grad|: 8.434912e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 1 NLL: 16.602760037

loto7_4710_k86.csv / validacija 2 / ridge 0.1
Trening prelaza: 3296
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 0.1
Iteracija 5/100 | cilj 16.510775443 | Max|proj.grad| 9.229e-03 | proteklo 0.5 s
Iteracija 10/100 | cilj 16.508061638 | Max|proj.grad| 2.323e-03 | proteklo 0.9 s
Iteracija 15/100 | cilj 16.507962882 | Max|proj.grad| 1.422e-03 | proteklo 1.4 s
Iteracija 20/100 | cilj 16.507945683 | Max|proj.grad| 7.684e-05 | proteklo 1.8 s
Iteracija 25/100 | cilj 16.507944329 | Max|proj.grad| 1.756e-05 | proteklo 2.3 s
Iteracija 30/100 | cilj 16.507944160 | Max|proj.grad| 2.579e-05 | proteklo 2.6 s
Iteracija 35/100 | cilj 16.507944005 | Max|proj.grad| 5.244e-05 | proteklo 3.0 s
Iteracija 40/100 | cilj 16.507943250 | Max|proj.grad| 3.178e-05 | proteklo 3.4 s
Iteracija 45/100 | cilj 16.507942832 | Max|proj.grad| 2.965e-05 | proteklo 3.8 s
Iteracija 50/100 | cilj 16.507942799 | Max|proj.grad| 9.888e-06 | proteklo 4.3 s
Iteracija 55/100 | cilj 16.507942794 | Max|proj.grad| 1.092e-06 | proteklo 4.7 s
Iteracija 60/100 | cilj 16.507942794 | Max|proj.grad| 9.161e-07 | proteklo 5.1 s
Zadržani cilj: 16.507942794
Max|proj.grad|: 9.160762e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 2 NLL: 16.591427945
Objedinjeni validacioni NLL za ridge 0.1: 16.597093991

loto7_4710_k86.csv / validacija 1 / ridge 1.0
Trening prelaza: 2825
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.504785597 | Max|proj.grad| 1.695e-02 | proteklo 0.4 s
Iteracija 10/100 | cilj 16.501852473 | Max|proj.grad| 2.491e-03 | proteklo 0.9 s
Iteracija 15/100 | cilj 16.501724344 | Max|proj.grad| 5.618e-04 | proteklo 1.3 s
Iteracija 20/100 | cilj 16.501713550 | Max|proj.grad| 9.052e-05 | proteklo 1.6 s
Iteracija 25/100 | cilj 16.501710449 | Max|proj.grad| 5.801e-05 | proteklo 2.1 s
Iteracija 30/100 | cilj 16.501708482 | Max|proj.grad| 6.434e-05 | proteklo 2.4 s
Iteracija 35/100 | cilj 16.501708288 | Max|proj.grad| 1.770e-05 | proteklo 2.8 s
Iteracija 40/100 | cilj 16.501708256 | Max|proj.grad| 3.366e-06 | proteklo 3.1 s
Iteracija 45/100 | cilj 16.501708252 | Max|proj.grad| 2.558e-06 | proteklo 3.5 s
Iteracija 50/100 | cilj 16.501708250 | Max|proj.grad| 1.572e-06 | proteklo 3.9 s
Zadržani cilj: 16.501708249
Max|proj.grad|: 6.567934e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 1 NLL: 16.596061495

loto7_4710_k86.csv / validacija 2 / ridge 1.0
Trening prelaza: 3296
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.512048578 | Max|proj.grad| 8.514e-03 | proteklo 0.4 s
Iteracija 10/100 | cilj 16.510105390 | Max|proj.grad| 1.575e-03 | proteklo 0.9 s
Iteracija 15/100 | cilj 16.510062683 | Max|proj.grad| 8.657e-04 | proteklo 1.4 s
Iteracija 20/100 | cilj 16.510055411 | Max|proj.grad| 1.136e-04 | proteklo 1.8 s
Iteracija 25/100 | cilj 16.510053372 | Max|proj.grad| 5.387e-05 | proteklo 2.1 s
Iteracija 30/100 | cilj 16.510052395 | Max|proj.grad| 2.158e-05 | proteklo 2.5 s
Iteracija 35/100 | cilj 16.510052352 | Max|proj.grad| 1.096e-05 | proteklo 2.9 s
Iteracija 40/100 | cilj 16.510052348 | Max|proj.grad| 2.657e-06 | proteklo 3.3 s
Iteracija 45/100 | cilj 16.510052346 | Max|proj.grad| 2.728e-06 | proteklo 3.7 s
Iteracija 50/100 | cilj 16.510052345 | Max|proj.grad| 8.450e-07 | proteklo 4.1 s
Zadržani cilj: 16.510052345
Max|proj.grad|: 8.450364e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 2 NLL: 16.588384639
Objedinjeni validacioni NLL za ridge 1.0: 16.592223067

loto7_4710_k86.csv: izabrani ridge 1.0
Validacioni NLL: 16.592223067

loto7_4710_k86.csv / model za odvojeni test
Trening prelaza: 3767
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.517760004 | Max|proj.grad| 1.312e-02 | proteklo 0.5 s
Iteracija 10/100 | cilj 16.515681504 | Max|proj.grad| 1.349e-03 | proteklo 1.0 s
Iteracija 15/100 | cilj 16.515646143 | Max|proj.grad| 7.491e-04 | proteklo 1.4 s
Iteracija 20/100 | cilj 16.515639978 | Max|proj.grad| 7.501e-05 | proteklo 1.8 s
Iteracija 25/100 | cilj 16.515637801 | Max|proj.grad| 4.709e-05 | proteklo 2.3 s
Iteracija 30/100 | cilj 16.515637095 | Max|proj.grad| 4.932e-05 | proteklo 2.7 s
Iteracija 35/100 | cilj 16.515637061 | Max|proj.grad| 1.278e-05 | proteklo 3.3 s
Iteracija 40/100 | cilj 16.515637056 | Max|proj.grad| 1.454e-06 | proteklo 3.7 s
Iteracija 45/100 | cilj 16.515637055 | Max|proj.grad| 1.685e-06 | proteklo 4.1 s
Zadržani cilj: 16.515637054
Max|proj.grad|: 9.005799e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86.csv / završno učenje na svim prelazima
Trening prelaza: 4709
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.523363388 | Max|proj.grad| 7.495e-03 | proteklo 0.6 s
Iteracija 10/100 | cilj 16.521985672 | Max|proj.grad| 1.306e-03 | proteklo 1.2 s
Iteracija 15/100 | cilj 16.521955544 | Max|proj.grad| 7.113e-04 | proteklo 1.6 s
Iteracija 20/100 | cilj 16.521952207 | Max|proj.grad| 1.215e-04 | proteklo 2.2 s
Iteracija 25/100 | cilj 16.521951606 | Max|proj.grad| 1.022e-04 | proteklo 2.6 s
Iteracija 30/100 | cilj 16.521951201 | Max|proj.grad| 1.557e-05 | proteklo 3.2 s
Iteracija 35/100 | cilj 16.521951184 | Max|proj.grad| 3.617e-06 | proteklo 3.7 s
Zadržani cilj: 16.521951183
Max|proj.grad|: 9.999664e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86.csv
NEXT: 8 10 14 22 23 34 37
Validacioni NLL: 16.592223067
Odvojeni test NLL: 16.579009426
Test NLL kontrole bez poruke: 16.548639443
Modelna verovatnoća NEXT: 1.50967925449e-07
Normalizacioni odnos: 1.000000000000
Greška dve normalizacije: 3.553e-15
Zbir marginalnih verovatnoća: 7.000000000000
Greška Qiskit komunikacionog izraza: 4.441e-16
Greška Bobovih marginala pre poruke: 3.331e-16
Najveće kalibraciono prekoračenje kapaciteta: 0.000e+00
Dijagnostička osetljivost: 6.852815e-02
Završni Max|proj.grad|: 9.999664e-07
Naučeni parovi ulaznih brojeva:
1-7 | 2-8 | 3-34 | 4-9 | 5-27 | 6-18 | 10-13 | 11-38 | 12-21 | 14-26 | 15-37 | 16-28 | 17-31 | 19-30 | 20-24 | 22-32 | 23: dva istorijska trenutka | 25-35 | 29-33 | 36-39
Test: broj kola prema broju pogodaka
0: 206 | 1: 371 | 2: 279 | 3: 78 | 4: 7 | 5: 1 | 6: 0 | 7: 0
Prosečan broj pogodaka MAP kombinacije: 1.269639
Marginalne verovatnoće uključivanja brojeva:
01: 0.17557051 | 02: 0.18853278 | 03: 0.18561601 | 04: 0.16547203 | 05: 0.18094295 | 06: 0.18489451
07: 0.18113124 | 08: 0.19536057 | 09: 0.17657393 | 10: 0.21774130 | 11: 0.17541293 | 12: 0.17988451
13: 0.15673373 | 14: 0.19848911 | 15: 0.14748757 | 16: 0.18580444 | 17: 0.16579082 | 18: 0.17430998
19: 0.17853436 | 20: 0.16645557 | 21: 0.17384494 | 22: 0.20433386 | 23: 0.19529223 | 24: 0.18737863
25: 0.16752931 | 26: 0.18275442 | 27: 0.17059586 | 28: 0.18119687 | 29: 0.18879487 | 30: 0.16524838
31: 0.17245976 | 32: 0.16204719 | 33: 0.18743247 | 34: 0.18952935 | 35: 0.17975358 | 36: 0.15311361
37: 0.19009254 | 38: 0.17923436 | 39: 0.18862893
TEST STATUS: POBOLJŠANJE NIJE POTVRĐENO
DIJAGNOSTIKA: Odvojeni test ne potvrđuje poboljšanje prema kontroli bez poruke.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.1
Trening prelaza: 1785
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 0.1
Iteracija 5/100 | cilj 16.473790206 | Max|proj.grad| 1.149e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.469603518 | Max|proj.grad| 2.244e-03 | proteklo 0.7 s
Iteracija 15/100 | cilj 16.469377711 | Max|proj.grad| 1.066e-03 | proteklo 0.9 s
Iteracija 20/100 | cilj 16.469350322 | Max|proj.grad| 1.487e-04 | proteklo 1.3 s
Iteracija 25/100 | cilj 16.469346644 | Max|proj.grad| 4.387e-05 | proteklo 1.6 s
Iteracija 30/100 | cilj 16.469345972 | Max|proj.grad| 4.675e-05 | proteklo 1.9 s
Iteracija 35/100 | cilj 16.469344825 | Max|proj.grad| 1.507e-04 | proteklo 2.1 s
Iteracija 40/100 | cilj 16.469342183 | Max|proj.grad| 1.064e-04 | proteklo 2.4 s
Iteracija 45/100 | cilj 16.469340804 | Max|proj.grad| 8.914e-05 | proteklo 2.8 s
Iteracija 50/100 | cilj 16.469340657 | Max|proj.grad| 1.921e-05 | proteklo 3.0 s
Iteracija 55/100 | cilj 16.469340637 | Max|proj.grad| 2.576e-06 | proteklo 3.3 s
Iteracija 60/100 | cilj 16.469340633 | Max|proj.grad| 2.416e-06 | proteklo 3.6 s
Iteracija 65/100 | cilj 16.469340629 | Max|proj.grad| 5.721e-06 | proteklo 3.9 s
Iteracija 70/100 | cilj 16.469340619 | Max|proj.grad| 8.835e-06 | proteklo 4.1 s
Iteracija 75/100 | cilj 16.469340604 | Max|proj.grad| 4.224e-06 | proteklo 4.4 s
Iteracija 80/100 | cilj 16.469340603 | Max|proj.grad| 1.993e-06 | proteklo 4.7 s
Zadržani cilj: 16.469340603
Max|proj.grad|: 6.198575e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 1 NLL: 16.584153747

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.1
Trening prelaza: 2083
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 0.1
Iteracija 5/100 | cilj 16.481173654 | Max|proj.grad| 1.308e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.476095984 | Max|proj.grad| 1.826e-03 | proteklo 0.7 s
Iteracija 15/100 | cilj 16.475884627 | Max|proj.grad| 4.445e-04 | proteklo 1.0 s
Iteracija 20/100 | cilj 16.475868196 | Max|proj.grad| 1.079e-04 | proteklo 1.3 s
Iteracija 25/100 | cilj 16.475866165 | Max|proj.grad| 4.387e-05 | proteklo 1.7 s
Iteracija 30/100 | cilj 16.475865623 | Max|proj.grad| 8.647e-05 | proteklo 2.0 s
Iteracija 35/100 | cilj 16.475864333 | Max|proj.grad| 1.303e-04 | proteklo 2.3 s
Iteracija 40/100 | cilj 16.475861885 | Max|proj.grad| 1.227e-04 | proteklo 2.6 s
Iteracija 45/100 | cilj 16.475861037 | Max|proj.grad| 4.873e-05 | proteklo 2.9 s
Iteracija 50/100 | cilj 16.475860976 | Max|proj.grad| 9.933e-06 | proteklo 3.2 s
Iteracija 55/100 | cilj 16.475860970 | Max|proj.grad| 1.891e-06 | proteklo 3.5 s
Iteracija 60/100 | cilj 16.475860968 | Max|proj.grad| 2.082e-06 | proteklo 3.8 s
Iteracija 65/100 | cilj 16.475860961 | Max|proj.grad| 9.563e-06 | proteklo 4.1 s
Iteracija 70/100 | cilj 16.475860948 | Max|proj.grad| 8.166e-06 | proteklo 4.4 s
Iteracija 75/100 | cilj 16.475860946 | Max|proj.grad| 1.685e-06 | proteklo 4.7 s
Zadržani cilj: 16.475860946
Max|proj.grad|: 8.895658e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 2 NLL: 16.634219404
Objedinjeni validacioni NLL za ridge 0.1: 16.609144504

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 1.0
Trening prelaza: 1785
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.476299277 | Max|proj.grad| 1.060e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.473188507 | Max|proj.grad| 1.758e-03 | proteklo 0.6 s
Iteracija 15/100 | cilj 16.473079408 | Max|proj.grad| 1.501e-03 | proteklo 0.9 s
Iteracija 20/100 | cilj 16.473056837 | Max|proj.grad| 1.329e-04 | proteklo 1.2 s
Iteracija 25/100 | cilj 16.473047471 | Max|proj.grad| 1.188e-04 | proteklo 1.5 s
Iteracija 30/100 | cilj 16.473045135 | Max|proj.grad| 5.060e-05 | proteklo 1.8 s
Iteracija 35/100 | cilj 16.473044953 | Max|proj.grad| 1.267e-05 | proteklo 2.1 s
Iteracija 40/100 | cilj 16.473044927 | Max|proj.grad| 7.429e-06 | proteklo 2.4 s
Iteracija 45/100 | cilj 16.473044914 | Max|proj.grad| 4.623e-06 | proteklo 2.7 s
Iteracija 50/100 | cilj 16.473044909 | Max|proj.grad| 1.856e-06 | proteklo 2.9 s
Zadržani cilj: 16.473044909
Max|proj.grad|: 8.236301e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 1 NLL: 16.578277857

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 1.0
Trening prelaza: 2083
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.483442179 | Max|proj.grad| 1.217e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.479738309 | Max|proj.grad| 1.343e-03 | proteklo 0.7 s
Iteracija 15/100 | cilj 16.479640048 | Max|proj.grad| 2.373e-04 | proteklo 1.0 s
Iteracija 20/100 | cilj 16.479629211 | Max|proj.grad| 1.217e-04 | proteklo 1.3 s
Iteracija 25/100 | cilj 16.479621103 | Max|proj.grad| 1.477e-04 | proteklo 1.6 s
Iteracija 30/100 | cilj 16.479618783 | Max|proj.grad| 3.586e-05 | proteklo 2.0 s
Iteracija 35/100 | cilj 16.479618625 | Max|proj.grad| 1.601e-05 | proteklo 2.3 s
Iteracija 40/100 | cilj 16.479618609 | Max|proj.grad| 1.392e-05 | proteklo 2.6 s
Iteracija 45/100 | cilj 16.479618597 | Max|proj.grad| 4.699e-06 | proteklo 2.9 s
Iteracija 50/100 | cilj 16.479618594 | Max|proj.grad| 1.500e-06 | proteklo 3.3 s
Zadržani cilj: 16.479618594
Max|proj.grad|: 5.520529e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 2 NLL: 16.624516004
Objedinjeni validacioni NLL za ridge 1.0: 16.601358075

loto7_4710_k86_loto_2977.csv: izabrani ridge 1.0
Validacioni NLL: 16.601358075

loto7_4710_k86_loto_2977.csv / model za odvojeni test
Trening prelaza: 2380
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.494413728 | Max|proj.grad| 8.297e-03 | proteklo 0.4 s
Iteracija 10/100 | cilj 16.491601761 | Max|proj.grad| 1.323e-03 | proteklo 0.7 s
Iteracija 15/100 | cilj 16.491541536 | Max|proj.grad| 6.649e-04 | proteklo 1.1 s
Iteracija 20/100 | cilj 16.491529663 | Max|proj.grad| 6.171e-04 | proteklo 1.4 s
Iteracija 25/100 | cilj 16.491522734 | Max|proj.grad| 1.547e-04 | proteklo 1.8 s
Iteracija 30/100 | cilj 16.491520727 | Max|proj.grad| 4.295e-05 | proteklo 2.2 s
Iteracija 35/100 | cilj 16.491520567 | Max|proj.grad| 9.355e-06 | proteklo 2.5 s
Iteracija 40/100 | cilj 16.491520554 | Max|proj.grad| 7.489e-06 | proteklo 2.8 s
Iteracija 45/100 | cilj 16.491520549 | Max|proj.grad| 4.506e-06 | proteklo 3.2 s
Iteracija 50/100 | cilj 16.491520546 | Max|proj.grad| 2.763e-06 | proteklo 3.6 s
Zadržani cilj: 16.491520546
Max|proj.grad|: 9.444541e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima
Trening prelaza: 2976
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.504957640 | Max|proj.grad| 9.020e-03 | proteklo 0.4 s
Iteracija 10/100 | cilj 16.503077910 | Max|proj.grad| 2.086e-03 | proteklo 0.9 s
Iteracija 15/100 | cilj 16.503002435 | Max|proj.grad| 3.788e-04 | proteklo 1.3 s
Iteracija 20/100 | cilj 16.502996983 | Max|proj.grad| 7.663e-05 | proteklo 1.6 s
Iteracija 25/100 | cilj 16.502994150 | Max|proj.grad| 7.591e-05 | proteklo 2.1 s
Iteracija 30/100 | cilj 16.502992975 | Max|proj.grad| 2.035e-05 | proteklo 2.4 s
Iteracija 35/100 | cilj 16.502992881 | Max|proj.grad| 8.870e-06 | proteklo 2.9 s
Iteracija 40/100 | cilj 16.502992873 | Max|proj.grad| 4.776e-06 | proteklo 3.3 s
Iteracija 45/100 | cilj 16.502992871 | Max|proj.grad| 6.898e-06 | proteklo 3.7 s
Iteracija 50/100 | cilj 16.502992869 | Max|proj.grad| 2.781e-06 | proteklo 4.1 s
Zadržani cilj: 16.502992869
Max|proj.grad|: 7.142730e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86_loto_2977.csv
NEXT: 9 22 24 28 29 34 35
Validacioni NLL: 16.601358075
Odvojeni test NLL: 16.585374892
Test NLL kontrole bez poruke: 16.548639443
Modelna verovatnoća NEXT: 2.51304499768e-07
Normalizacioni odnos: 1.000000000000
Greška dve normalizacije: 7.105e-15
Zbir marginalnih verovatnoća: 7.000000000000
Greška Qiskit komunikacionog izraza: 4.441e-16
Greška Bobovih marginala pre poruke: 3.331e-16
Najveće kalibraciono prekoračenje kapaciteta: 0.000e+00
Dijagnostička osetljivost: 8.525110e-02
Završni Max|proj.grad|: 7.142730e-07
Naučeni parovi ulaznih brojeva:
1-4 | 2-3 | 5-36 | 6-34 | 7-39 | 8-18 | 9-28 | 10-33 | 11-27 | 12-15 | 13-21 | 14-25 | 16-26 | 17-20 | 19-37 | 22: dva istorijska trenutka | 23-30 | 24-32 | 29-35 | 31-38
Test: broj kola prema broju pogodaka
0: 124 | 1: 251 | 2: 163 | 3: 46 | 4: 12 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka MAP kombinacije: 1.280201
Marginalne verovatnoće uključivanja brojeva:
01: 0.16515455 | 02: 0.16164044 | 03: 0.17802364 | 04: 0.16002251 | 05: 0.17216163 | 06: 0.18213630
07: 0.17034350 | 08: 0.19395369 | 09: 0.20731626 | 10: 0.18097120 | 11: 0.17719633 | 12: 0.16222031
13: 0.18063970 | 14: 0.15680361 | 15: 0.14299105 | 16: 0.18813365 | 17: 0.15903057 | 18: 0.17582868
19: 0.18033536 | 20: 0.16758551 | 21: 0.16932793 | 22: 0.20129170 | 23: 0.18378815 | 24: 0.21663624
25: 0.16431589 | 26: 0.18625241 | 27: 0.15246308 | 28: 0.21648426 | 29: 0.20350052 | 30: 0.17665835
31: 0.17036934 | 32: 0.17451647 | 33: 0.17596775 | 34: 0.22898464 | 35: 0.20263535 | 36: 0.17098132
37: 0.18723288 | 38: 0.18061358 | 39: 0.17549166
TEST STATUS: POBOLJŠANJE NIJE POTVRĐENO
DIJAGNOSTIKA: Odvojeni test ne potvrđuje poboljšanje prema kontroli bez poruke.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.1
Trening prelaza: 1039
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 0.1
Iteracija 5/100 | cilj 16.429325767 | Max|proj.grad| 1.983e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.419358602 | Max|proj.grad| 2.741e-03 | proteklo 0.6 s
Iteracija 15/100 | cilj 16.417791143 | Max|proj.grad| 4.160e-03 | proteklo 0.9 s
Iteracija 20/100 | cilj 16.417403029 | Max|proj.grad| 1.213e-03 | proteklo 1.1 s
Iteracija 25/100 | cilj 16.417233877 | Max|proj.grad| 1.436e-03 | proteklo 1.4 s
Iteracija 30/100 | cilj 16.417157313 | Max|proj.grad| 2.551e-04 | proteklo 1.7 s
Iteracija 35/100 | cilj 16.417134567 | Max|proj.grad| 3.325e-04 | proteklo 1.9 s
Iteracija 40/100 | cilj 16.417129976 | Max|proj.grad| 1.607e-04 | proteklo 2.2 s
Iteracija 45/100 | cilj 16.417127951 | Max|proj.grad| 1.035e-04 | proteklo 2.4 s
Iteracija 50/100 | cilj 16.417124925 | Max|proj.grad| 1.249e-04 | proteklo 2.6 s
Iteracija 55/100 | cilj 16.417118659 | Max|proj.grad| 1.885e-04 | proteklo 2.9 s
Iteracija 60/100 | cilj 16.417114107 | Max|proj.grad| 1.249e-04 | proteklo 3.1 s
Iteracija 65/100 | cilj 16.417112160 | Max|proj.grad| 1.538e-04 | proteklo 3.4 s
Iteracija 70/100 | cilj 16.417110264 | Max|proj.grad| 9.349e-05 | proteklo 3.6 s
Iteracija 75/100 | cilj 16.417109044 | Max|proj.grad| 6.608e-05 | proteklo 3.8 s
Iteracija 80/100 | cilj 16.417108559 | Max|proj.grad| 3.808e-05 | proteklo 4.1 s
Iteracija 85/100 | cilj 16.417108056 | Max|proj.grad| 6.405e-05 | proteklo 4.4 s
Iteracija 90/100 | cilj 16.417106901 | Max|proj.grad| 1.241e-04 | proteklo 4.6 s
Iteracija 95/100 | cilj 16.417105195 | Max|proj.grad| 5.239e-05 | proteklo 4.8 s
Iteracija 100/100 | cilj 16.417104711 | Max|proj.grad| 3.995e-05 | proteklo 5.0 s
Zadržani cilj: 16.417104711
Max|proj.grad|: 3.995096e-05
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Validacija 1 NLL: 16.712622428

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.1
Trening prelaza: 1212
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 0.1
Iteracija 5/100 | cilj 16.443155203 | Max|proj.grad| 1.942e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.434020457 | Max|proj.grad| 2.451e-03 | proteklo 0.5 s
Iteracija 15/100 | cilj 16.433260569 | Max|proj.grad| 9.516e-04 | proteklo 0.8 s
Iteracija 20/100 | cilj 16.433123911 | Max|proj.grad| 4.603e-04 | proteklo 1.0 s
Iteracija 25/100 | cilj 16.433091383 | Max|proj.grad| 3.463e-04 | proteklo 1.3 s
Iteracija 30/100 | cilj 16.433084864 | Max|proj.grad| 6.052e-05 | proteklo 1.5 s
Iteracija 35/100 | cilj 16.433083152 | Max|proj.grad| 8.826e-05 | proteklo 1.8 s
Iteracija 40/100 | cilj 16.433081206 | Max|proj.grad| 8.483e-05 | proteklo 2.0 s
Iteracija 45/100 | cilj 16.433077605 | Max|proj.grad| 2.512e-04 | proteklo 2.3 s
Iteracija 50/100 | cilj 16.433073708 | Max|proj.grad| 5.317e-05 | proteklo 2.5 s
Iteracija 55/100 | cilj 16.433073203 | Max|proj.grad| 2.922e-05 | proteklo 2.8 s
Iteracija 60/100 | cilj 16.433073125 | Max|proj.grad| 9.379e-06 | proteklo 3.0 s
Iteracija 65/100 | cilj 16.433073094 | Max|proj.grad| 1.533e-05 | proteklo 3.3 s
Iteracija 70/100 | cilj 16.433073056 | Max|proj.grad| 1.751e-05 | proteklo 3.5 s
Iteracija 75/100 | cilj 16.433072973 | Max|proj.grad| 5.261e-05 | proteklo 3.8 s
Iteracija 80/100 | cilj 16.433072915 | Max|proj.grad| 6.797e-06 | proteklo 4.1 s
Iteracija 85/100 | cilj 16.433072904 | Max|proj.grad| 3.814e-06 | proteklo 4.3 s
Iteracija 90/100 | cilj 16.433072900 | Max|proj.grad| 3.713e-06 | proteklo 4.5 s
Iteracija 95/100 | cilj 16.433072898 | Max|proj.grad| 2.900e-06 | proteklo 4.8 s
Iteracija 100/100 | cilj 16.433072895 | Max|proj.grad| 5.361e-06 | proteklo 5.1 s
Zadržani cilj: 16.433072895
Max|proj.grad|: 5.360936e-06
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Validacija 2 NLL: 16.706683541
Objedinjeni validacioni NLL za ridge 0.1: 16.709652985

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 1.0
Trening prelaza: 1039
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.433377362 | Max|proj.grad| 1.830e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.426089793 | Max|proj.grad| 1.976e-03 | proteklo 0.6 s
Iteracija 15/100 | cilj 16.425453165 | Max|proj.grad| 1.838e-03 | proteklo 0.8 s
Iteracija 20/100 | cilj 16.425365162 | Max|proj.grad| 1.386e-03 | proteklo 1.0 s
Iteracija 25/100 | cilj 16.425335857 | Max|proj.grad| 1.818e-04 | proteklo 1.3 s
Iteracija 30/100 | cilj 16.425328060 | Max|proj.grad| 3.361e-04 | proteklo 1.5 s
Iteracija 35/100 | cilj 16.425325075 | Max|proj.grad| 8.446e-05 | proteklo 1.8 s
Iteracija 40/100 | cilj 16.425323974 | Max|proj.grad| 3.369e-05 | proteklo 2.0 s
Iteracija 45/100 | cilj 16.425323697 | Max|proj.grad| 1.740e-05 | proteklo 2.3 s
Iteracija 50/100 | cilj 16.425323599 | Max|proj.grad| 4.574e-05 | proteklo 2.5 s
Iteracija 55/100 | cilj 16.425323541 | Max|proj.grad| 2.452e-05 | proteklo 2.7 s
Iteracija 60/100 | cilj 16.425323527 | Max|proj.grad| 1.288e-05 | proteklo 2.9 s
Iteracija 65/100 | cilj 16.425323525 | Max|proj.grad| 8.346e-06 | proteklo 3.2 s
Iteracija 70/100 | cilj 16.425323523 | Max|proj.grad| 4.100e-06 | proteklo 3.4 s
Iteracija 75/100 | cilj 16.425323523 | Max|proj.grad| 6.180e-07 | proteklo 3.7 s
Zadržani cilj: 16.425323523
Max|proj.grad|: 6.179548e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 1 NLL: 16.666438439

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 1.0
Trening prelaza: 1212
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.446530917 | Max|proj.grad| 1.796e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.439769317 | Max|proj.grad| 1.889e-03 | proteklo 0.5 s
Iteracija 15/100 | cilj 16.439444096 | Max|proj.grad| 5.572e-04 | proteklo 0.8 s
Iteracija 20/100 | cilj 16.439399140 | Max|proj.grad| 3.413e-04 | proteklo 1.0 s
Iteracija 25/100 | cilj 16.439377564 | Max|proj.grad| 3.284e-04 | proteklo 1.3 s
Iteracija 30/100 | cilj 16.439371929 | Max|proj.grad| 8.941e-05 | proteklo 1.5 s
Iteracija 35/100 | cilj 16.439370265 | Max|proj.grad| 3.369e-05 | proteklo 1.7 s
Iteracija 40/100 | cilj 16.439370090 | Max|proj.grad| 2.492e-05 | proteklo 2.0 s
Iteracija 45/100 | cilj 16.439370044 | Max|proj.grad| 2.831e-05 | proteklo 2.2 s
Iteracija 50/100 | cilj 16.439370015 | Max|proj.grad| 6.635e-06 | proteklo 2.5 s
Iteracija 55/100 | cilj 16.439370008 | Max|proj.grad| 1.655e-06 | proteklo 2.7 s
Zadržani cilj: 16.439370007
Max|proj.grad|: 9.557918e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
Validacija 2 NLL: 16.693125470
Objedinjeni validacioni NLL za ridge 1.0: 16.679781954

loto7_4710_k86_loto_plus_1733.csv: izabrani ridge 1.0
Validacioni NLL: 16.679781954

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test
Trening prelaza: 1385
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.466470849 | Max|proj.grad| 1.892e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.459367457 | Max|proj.grad| 3.005e-03 | proteklo 0.7 s
Iteracija 15/100 | cilj 16.458808281 | Max|proj.grad| 8.600e-04 | proteklo 0.9 s
Iteracija 20/100 | cilj 16.458712364 | Max|proj.grad| 3.170e-04 | proteklo 1.3 s
Iteracija 25/100 | cilj 16.458690116 | Max|proj.grad| 1.095e-04 | proteklo 1.6 s
Iteracija 30/100 | cilj 16.458684899 | Max|proj.grad| 7.519e-05 | proteklo 1.8 s
Iteracija 35/100 | cilj 16.458683687 | Max|proj.grad| 4.319e-05 | proteklo 2.0 s
Iteracija 40/100 | cilj 16.458683291 | Max|proj.grad| 1.832e-05 | proteklo 2.3 s
Iteracija 45/100 | cilj 16.458683216 | Max|proj.grad| 1.151e-05 | proteklo 2.6 s
Iteracija 50/100 | cilj 16.458683191 | Max|proj.grad| 1.320e-05 | proteklo 2.8 s
Iteracija 55/100 | cilj 16.458683186 | Max|proj.grad| 1.827e-06 | proteklo 3.1 s
Zadržani cilj: 16.458683185
Max|proj.grad|: 7.570574e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima
Trening prelaza: 1732
Kvantnih protokola: 20
Mernih parametara: 240
Regularizacija: 1.0
Iteracija 5/100 | cilj 16.478975019 | Max|proj.grad| 1.869e-02 | proteklo 0.3 s
Iteracija 10/100 | cilj 16.473837974 | Max|proj.grad| 1.599e-03 | proteklo 0.7 s
Iteracija 15/100 | cilj 16.473639016 | Max|proj.grad| 3.080e-04 | proteklo 0.9 s
Iteracija 20/100 | cilj 16.473618301 | Max|proj.grad| 4.501e-04 | proteklo 1.2 s
Iteracija 25/100 | cilj 16.473608559 | Max|proj.grad| 1.950e-04 | proteklo 1.5 s
Iteracija 30/100 | cilj 16.473604912 | Max|proj.grad| 7.618e-05 | proteklo 1.8 s
Iteracija 35/100 | cilj 16.473604459 | Max|proj.grad| 1.045e-04 | proteklo 2.1 s
Iteracija 40/100 | cilj 16.473604377 | Max|proj.grad| 7.222e-06 | proteklo 2.3 s
Iteracija 45/100 | cilj 16.473604357 | Max|proj.grad| 6.875e-06 | proteklo 2.6 s
Iteracija 50/100 | cilj 16.473604347 | Max|proj.grad| 4.228e-06 | proteklo 2.9 s
Iteracija 55/100 | cilj 16.473604344 | Max|proj.grad| 1.858e-06 | proteklo 3.1 s
Zadržani cilj: 16.473604344
Max|proj.grad|: 3.378483e-06
Status optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH

loto7_4710_k86_loto_plus_1733.csv
NEXT: 2 5 10 23 29 34 39
Validacioni NLL: 16.679781954
Odvojeni test NLL: 16.617279686
Test NLL kontrole bez poruke: 16.548639443
Modelna verovatnoća NEXT: 1.96901750296e-07
Normalizacioni odnos: 1.000000000000
Greška dve normalizacije: 0.000e+00
Zbir marginalnih verovatnoća: 7.000000000000
Greška Qiskit komunikacionog izraza: 5.551e-16
Greška Bobovih marginala pre poruke: 3.886e-16
Najveće kalibraciono prekoračenje kapaciteta: 0.000e+00
Dijagnostička osetljivost: 1.065227e-01
Završni Max|proj.grad|: 3.378483e-06
Naučeni parovi ulaznih brojeva:
1-39 | 2-6 | 3-13 | 4-29 | 5-21 | 7-14 | 8-11 | 9-24 | 10-16 | 12: dva istorijska trenutka | 15-17 | 18-36 | 19-23 | 20-30 | 22-37 | 25-27 | 26-33 | 28-35 | 31-32 | 34-38
Test: broj kola prema broju pogodaka
0: 71 | 1: 132 | 2: 108 | 3: 32 | 4: 4 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka MAP kombinacije: 1.325648
Marginalne verovatnoće uključivanja brojeva:
01: 0.16474077 | 02: 0.20642539 | 03: 0.11871555 | 04: 0.19271515 | 05: 0.19440667 | 06: 0.18194217
07: 0.18247144 | 08: 0.17860098 | 09: 0.17410056 | 10: 0.20056434 | 11: 0.19175755 | 12: 0.17564520
13: 0.14916805 | 14: 0.17976163 | 15: 0.17198505 | 16: 0.18866131 | 17: 0.15818592 | 18: 0.16495889
19: 0.16550597 | 20: 0.18951681 | 21: 0.17410313 | 22: 0.18519306 | 23: 0.21998503 | 24: 0.18746034
25: 0.17963654 | 26: 0.14514525 | 27: 0.19024222 | 28: 0.19315372 | 29: 0.20239035 | 30: 0.16990988
31: 0.18754091 | 32: 0.19222614 | 33: 0.15171104 | 34: 0.21821825 | 35: 0.17935128 | 36: 0.14841763
37: 0.16418850 | 38: 0.18741974 | 39: 0.19387760
TEST STATUS: POBOLJŠANJE NIJE POTVRĐENO
DIJAGNOSTIKA: Odvojeni test ne potvrđuje poboljšanje prema kontroli bez poruke.

ZAVRŠNI NEXT REZULTATI
loto7_4710_k86.csv | NEXT: 8 x 14 y 23 z 37 | TEST POBOLJŠANJE NIJE POTVRĐENO
loto7_4710_k86_loto_2977.csv | NEXT: 9 x 24 y 29 z 35 | TEST POBOLJŠANJE NIJE POTVRĐENO
loto7_4710_k86_loto_plus_1733.csv | NEXT: 2 x 10 y 29 z 39 | TEST POBOLJŠANJE NIJE POTVRĐENO
"""



"""
Kontrola količine informacije koja prolazi kroz Alice-Bob kanal. 

Na slici 2:
- Četiri optička kanala predstavljaju polarizacije dva fotona — to nisu četiri kubita.
- Alice prema svoja dva bita podešava mernu bazu.
- Bob prema traženom indeksu podešava svoju bazu.
- Alice šalje jednu klasičnu poruku; Bob je kombinuje sa svojim rezultatom.

Grana sa q = 1 može se predstaviti Qiskit kolom sa dva kubita. 
Grane sa q > 1 predstavljaju hipotetičke korelacije izvan kvantne granice; njih ne smem predstavljati kao regularno kvantno kolo. 

Zamena velike piramide izvora jednim izvorom i kanalom ograničenog kapaciteta. 
To je glavna ideja Miklinovog i Pawłowskog rada.

Za moj loto najkorisnije je:
1. Informaciono usko grlo: proveriti koliko istorijske informacije kvantni kod stvarno koristi za predikciju.
2. Provera bez signalizacije: Bobova raspodela pre prijema poruke ne sme zavisiti od Aliceinog lokalnog izbora.
3. Tačno sabiranje verovatnoća: notebook obračunava efekat greške kanala analitički; nije potrebno slučajno uzorkovanje.

Ne da se prenosi Alicein poznati bitovi, za Loto bi Alice smela da dobije samo prethodna kola. 
Buduće izvlačenje ostaje cilj učenja. Takođe, formula za informacionu količinu iz simetričnog dvobitnog eksperimenta ne može se neposredno preneti na zavisne loto-bitove.
Alice-Bob modela. 

V1 protokol sa slike 2: dva istorijska bita, dva spregnuta kubita, jedna Aliceina poruka i Bobov upit. 
Za 39 brojeva koristim odvojene protokole po parovima; to znači jednu poruku po paru, ne jedan bit za svih 39 brojeva. 
Predikcija će koristiti kanal bez šuma i q = 1. 
Kapacitet kanala proveravaću analitički, bez dodavanja šuma CSV podacima.

Dve odvojene provere: Bobove verovatnoće pre prijema poruke moraju ostati nezavisne od Aliceinog izbora, a informaciona kauzalnost proverava se na standardnom eksperimentu sa nezavisnim bitovima. 
Formulu iz tog eksperimenta neću primeniti direktno na zavisne loto-bitove. 
Predikcija će se učiti i oceniti preko verovatnoće cele sedmorke.

V1 koristi stvarni dvokubitni Alice-Bob protokol, q = 1, naučene merne uglove i jednu poruku po paru. 
Predikcija koristi kanal bez šuma. Provere kapaciteta rade analitički na zasebnom kalibracionom eksperimentu.
Matematički sam proverio idealnu rekonstrukciju poznatih bitova i kapacitet kanala.
"""
