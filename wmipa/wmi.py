"""This module implements the Weighted Model Integration calculation.

The calculation leverages:
    - a Satisfiability Modulo Theories solver supporting All-SMT (e.g. MathSAT)
    - a software computing exact volume of polynomials (e.g. LattE Integrale)

"""

__version__ = "1.0"
__author__ = "Gabriele Masina, Paolo Morettin, Giuseppe Spallitta"

import mathsat
import numpy as np
from pysmt.shortcuts import And, Bool, Iff, Not, Real, Solver, substitute, get_atoms
from pysmt.typing import BOOL, REAL

from wmipa.integration import LatteIntegrator
from wmipa.integration.integrator import Integrator
from wmipa.log import logger
from wmipa.utils import TermNormalizer, BooleanSimplifier
from wmipa.weights import Weights
from wmipa.weightconverter import WeightConverterSkeleton
from wmipa.wmiexception import WMIParsingException, WMIRuntimeException
from wmipa.wmivariables import WMIVariables


class WMISolver:
    """The class that has the purpose to calculate the Weighted Module Integration of
        a given support, weight function and query.

    Attributes:
        variables (WMIVariables): The list of variables created and used by WMISolver.
        weights (Weights): The representation of the weight function.
        chi (FNode): The pysmt formula that contains the support of the formula
        integrator (Integrator or list(Integrator)): The integrator or the list of integrators to use.
        simplifier (BooleanSimplifier): The class that simplifies the formula.
        normalizer (TermNormalizer): The class that normalizes LRA atoms.

    """

    def __init__(self, chi, w=Real(1), integrator=None):
        """Default constructor.

        Args:
            chi (FNode): The support of the problem.
            w (FNode, optional): The weight function of the problem (default: 1).
            integrator (Integrator or list(Integrator)): integrator or list of integrators to use. If a list of
                integrators is provided, then computeWMI will return a list of results, one for each integrator.
                (default: LatteIntegrator())

        """
        self.variables = WMIVariables()
        self.normalizer = TermNormalizer()
        self.weights = Weights(w)
        converterSK = WeightConverterSkeleton(self.variables)
        self.weights_as_formula_sk = converterSK.convert(w)
        self.chi = chi
        self.normalizer = TermNormalizer()
        self.integrator = integrator if integrator is not None else LatteIntegrator()
        self.simplifier = BooleanSimplifier()


    def computeWMI(self, phi, domain, cache=-1):

        """Calculates the WMI on a single query.

        Args:
            phi (FNode): The query on which to calculate the WMI.
            domain (set(FNode)): set of pysmt REAL vars encoding the integration domain
            cache (int, optional): The cache level to use when calculating WMI (default: -1 = no cache).

        Returns:
            real or np.ndarray(real): The result of the computation. If a list of integrators is provided, then the
                result is a np.ndarray(real) containing the results computed by the different integrators.
            int or np.ndarray(real): The number of integrations that have been computed. If a list of integrators is
                provided, then the result is a np.ndarray(int) containing the number of integrations computed by the
                different integrators.

        """

        logger.debug(f"Computing WMI (integration domain: {domain})")

        # domain of integration
        self.domain = domain

        # conjoin query and support
        formula = And(phi, self.chi)

        # sort the different atoms
        atoms = get_atoms(formula) | self.weights.get_atoms()
        bool_atoms = {a for a in atoms if a.is_symbol(BOOL)}
        lra_atoms = {a for a in atoms if a.is_theory_relation()}

        # conjoin the skeleton of the weight function
        formula = And(formula, self.weights_as_formula_sk)

        n_unassigned_bools = [] # keep track of n. of unassigned Boolean atoms
        convex_integrals = []
        if len(bool_atoms) == 0:
            # no Boolean atoms -> enumerate *partial* TAs over LRA atoms only
            for ta_lra in self._get_allsat(formula, lra_atoms):
                convex_integrals.append(self._assignment_to_integral(ta_lra))
                n_unassigned_bools.append(0)

        else:
            # enumerate *partial* TAs over Boolean atoms first
            for ta_bool in self._get_allsat(formula, bool_atoms):

                # dict containing all necessary truth values
                ta = dict(ta_bool)

                # try to simplify the formula using the partial TA
                is_convex, simplified_formula = self._simplify_formula(
                    formula, ta_bool, ta
                )

                if is_convex:
                    # simplified formula is a conjuction of atoms (we're done)
                    convex_integrals.append(self._assignment_to_integral(ta))
                    n_unassigned_bools.append(len(bool_atoms - ta_bool.keys()))

                else:
                    # simplified formula is non-covex, requiring another enumeration pass
                    residual_atoms = list({a for a in simplified_formula.get_free_variables()
                                           if a.symbol_type() == BOOL and a in bool_atoms})                    
                    residual_atoms.extend(list({a for a in simplified_formula.get_atoms()
                                                if a.is_theory_relation()}))

                    # may be both on LRA and boolean atoms
                    for ta_residual in self._get_allsat(simplified_formula, residual_atoms):
                        curr_ta = dict(ta)
                        curr_ta.update(ta_residual)
                        convex_integrals.append(self._assignment_to_integral(curr_ta))
                        n_unassigned_bools.append(len(bool_atoms - curr_ta.keys()))

        # multiply each volume by 2^(|A| - |mu^A|)
        factors = [2 ** nb for nb in n_unassigned_bools]
        volume, n_cached = self._integrate_batch(convex_integrals, cache, factors)
        n_integrations = len(convex_integrals) - n_cached
        
        logger.debug(f"Volume: {volume}, n_integrations: {n_integrations}, n_cached: {n_cached}")

        return volume, n_integrations


    def _integrate_batch(self, convex_integrals, cache, factors=None):
        """Computes the integral of a batch of convex_integrals.

        Args:
            convex_integrals (list): The list of convex_integrals to integrate.
            cache (int): The cache level to use.
            factors (list, optional): A list of factor each problem should be multiplied by.

        """
        if factors is None:
            factors = [1] * len(convex_integrals)
        else:
            assert isinstance(factors, list)
            assert len(convex_integrals) == len(factors)
        if isinstance(self.integrator, Integrator):
            results, cached = self.integrator.integrate_batch(convex_integrals, cache)
        else:
            results, cached = zip(*(i.integrate_batch(convex_integrals, cache) for i in self.integrator))
        cached = np.array(cached)
        results = np.array(results)
        volume = np.sum(results * factors, axis=-1)
        return volume, cached


    def _assignment_to_integral(self, atom_assignments):
        """Create a tuple containing the problem to integrate.

        It first finds all the aliases in the atom_assignments, then it takes the
            actual weight (based on the assignment).
        Finally, it creates the problem tuple with all the info in it.

        Args:
            atom_assignments (dict): Maps atoms to the corresponding truth value (True, False)

        Returns:
            tuple: The problem on which to calculate the integral formed by
                (atom assignment, actual weight, list of aliases, weight condition assignments)

        """
        aliases = {}
        for atom, value in atom_assignments.items():
            if value is True and atom.is_equals():
                alias, expr = self._parse_alias(atom)
                if self.variables.is_weight_alias(alias):
                    continue

                # check that there are no multiple assignments of the same alias
                if alias not in aliases:
                    aliases[alias] = expr
                else:
                    raise WMIParsingException(WMIParsingException.MULTIPLE_ASSIGNMENT_SAME_ALIAS)


        current_weight = self.weights.weight_from_assignment(atom_assignments)
        return atom_assignments, current_weight, aliases

    def _parse_alias(self, equality):
        """Takes an equality and parses it.

        Args:
            equality (FNode): The equality to parse.

        Returns:
            alias (FNode): The name of the alias.
            expr (FNode): The value of the alias.

        Raises:
            WMIParsingException: If the equality is not of the type
                (Symbol = real_formula) or vice-versa.

        """
        assert equality.is_equals(), "Not an equality"
        left, right = equality.args()
        if left.is_symbol() and (left.get_type() == REAL):
            alias, expr = left, right
        elif right.is_symbol() and (right.get_type() == REAL):
            alias, expr = right, left
        else:
            raise WMIParsingException(
                WMIParsingException.MALFORMED_ALIAS_EXPRESSION, equality
            )
        return alias, expr


    def _get_allsat(self, formula, atoms, force_total=False):

        """
        Gets the list of assignments that satisfy the formula.

        Args:
            formula (FNode): The formula to satisfy
            atoms (list): List of atoms on which to find the assignments.
            force_total (bool, optional): Forces total truth assignements.
                Defaults to False.

        Yields:
            list: assignments on the atoms
        """

        def _callback(model, converter, result):
            result.append([converter.back(v) for v in model])
            return 1

        msat_options = {
                "dpll.allsat_minimize_model": "true",
                "dpll.allsat_allow_duplicates": "false",
                "preprocessor.toplevel_propagation": "false",
                "preprocessor.simplification": "0",
            } if not force_total else {}

        # The current version of MathSAT returns a truth assignment on some
        # normalized version of the atoms instead of the original ones.
        # However, in order to simply get the value of the weight function
        # given a truth assignment, we need to know the truth assignment on
        # the original atoms.
        for atom in atoms:
            if not atom.is_symbol(BOOL):
                _ = self.normalizer.normalize(atom, remember_alias=True)

        solver = Solver(name="msat", solver_options=msat_options)
        converter = solver.converter
        solver.add_assertion(formula)

        # the MathSAT call returns models as conjunction of literals
        models = []
        mathsat.msat_all_sat(
            solver.msat_env(),
            [converter.convert(v) for v in atoms],
            lambda model: _callback(model, converter, models),
        )

        # convert each conjunction of literals to a dict {atoms : bool}
        for model in models:
            assignments = {}
            for lit in model:
                atom = lit.arg(0) if lit.is_not() else lit
                value = not lit.is_not()

                if atom.is_symbol(BOOL):
                    assignments[atom] = value
                else:
                    # retrieve the original (unnormalized) atom
                    normalized_atom, negated = self.normalizer.normalize(atom)
                    if negated:
                        value = not value
                    known_aliases = self.normalizer.known_aliases(normalized_atom)
                    for original_atom, negated in known_aliases:
                        assignments[original_atom] = (not value if negated else value)

            yield assignments

    def _simplify_formula(self, formula, subs, atom_assignments):
        """Substitute the subs in the formula and iteratively simplify it.
        atom_assignments is updated with unit-propagated atoms.

        Args:
            formula (FNode): The formula to simplify.
            subs (dict): Dictionary with the substitutions to perform.
            atom_assignments (dict): Dictionary with atoms and assigned value.

        Returns:
            bool: True if the formula is completely simplified.
            FNode: The simplified formula.
        """
        subs = {k: Bool(v) for k, v in subs.items()}
        f_next = formula
        # iteratively simplify F[A<-mu^A], getting (possibly part.) mu^LRA
        while True:
            f_before = f_next
            f_next = self.simplifier.simplify(substitute(f_before, subs))
            lra_assignments, is_convex = WMISolver._plra_rec(f_next, True)
            subs = {k: Bool(v) for k, v in lra_assignments.items()}
            atom_assignments.update(lra_assignments)
            if is_convex or lra_assignments == {}:
                break

        if not is_convex:
            # formula not completely simplified, add conjunction of assigned LRA atoms
            expressions = []
            for k, v in atom_assignments.items():
                if k.is_theory_relation():
                    if v:
                        expressions.append(k)
                    else:
                        expressions.append(Not(k))
            f_next = And([f_next] + expressions)
        return is_convex, f_next


    @staticmethod
    def _plra_rec(formula, pos_polarity):
        """This method extract all sub formulas in the formula and returns them as a dictionary.

        Args:
            formula (FNode): The formula to parse.
            pos_polarity (bool): The polarity of the formula.

        Returns:
            dict: the list of FNode in the formula with the corresponding truth value.
            bool: boolean that indicates if there are no more truth assignment to
                extract.

        """
        if formula.is_bool_constant():
            return {}, True
        elif formula.is_theory_relation() or formula.is_symbol(BOOL):
            return {formula: pos_polarity}, True
        elif formula.is_not():
            return WMISolver._plra_rec(formula.arg(0), not pos_polarity)
        elif formula.is_and() and pos_polarity:
            assignments = {}
            is_convex = True
            for a in formula.args():
                assignment, rec_is_convex = WMISolver._plra_rec(a, True)
                assignments.update(assignment)
                is_convex = rec_is_convex and is_convex
            return assignments, is_convex
        elif formula.is_or() and not pos_polarity:
            assignments = {}
            is_convex = True
            for a in formula.args():
                assignment, rec_is_convex = WMISolver._plra_rec(a, False)
                assignments.update(assignment)
                is_convex = rec_is_convex and is_convex
            return assignments, is_convex
        elif formula.is_implies() and not pos_polarity:
            assignments, is_convex_left = WMISolver._plra_rec(formula.arg(0), True)
            assignment_right, is_convex_right = WMISolver._plra_rec(formula.arg(1), False)
            assignments.update(assignment_right)
            return assignments, is_convex_left and is_convex_right
        else:
            return {}, False
