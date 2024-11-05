import pickle
from tempfile import TemporaryDirectory

from wmipa.integration.latte_integrator import LatteIntegrator
from wmipa.integration.torch.wmipa.numerical_symb_integrator_pa import NumericalSymbIntegratorPA
import torch
from scipy.spatial import ConvexHull
import os
import numpy as np
from subprocess import call

import tqdm


def write_polytope_v_file(vertices : np.ndarray, file_path: str):
    # format:
    # k (n + 1)
    # homogenized v_0
    # homogenized v_1
    # ...
    # homogenized v_n

    # homogenized v_i = (1, *v_i)
    # then gcd so all entries are integers

    n = vertices.shape[1]
    k = vertices.shape[0]

    with open(file_path, 'w') as file:
        file.write(f"{k} {n + 1}\n")
        for i in range(k):
            v = np.concatenate(([1], vertices[i]))
            mask = (v % 1) == 0
            if not mask.all():
                gcd = np.gcd.reduce(v)
                v = v / gcd
            file.write(" ".join(map(lambda x: str(int(x)), v)) + "\n")

def test_unstable_problem():
    total_degree = 60
    varmap = {'x_0': 0, 'x_1': 1, 'x_2': 2, 'x_3': 3}
    integrator = NumericalSymbIntegratorPA(
        total_degree=60,
        variable_map=varmap,
        batch_size=1024,
        n_workers=10,
        monomials_lower_precision=False,
    )

    with open("test/unit/problems.pkl", 'rb') as file:
        problems = pickle.load(file)

    integrator.set_device(torch.device("cuda:0"))

    problem = problems[0]

    atom_assignments, weight, aliases, cond_assignments = problem

    assert len(cond_assignments) == 0

    simplices, coeffs, exponents = integrator._convert_to_problem(atom_assignments, weight, aliases)

    assert exponents.sum(-1).max() <= total_degree
    
    simplices = simplices.to(integrator.device)
    coeffs = coeffs.to(integrator.device)
    exponents = exponents.to(integrator.device)
    integral_simplices = torch.vmap(
        lambda s: integrator.integrate_simplex(s, coeffs, exponents)
    )(simplices)
    integral_polytope = (
        torch.sum(integral_simplices, dim=0).to("cpu").unsqueeze(-1)
    )
    result = integral_polytope.item()

    latte = LatteIntegrator(n_threads=1)

    integrand, polytope = latte._convert_to_problem(
        atom_assignments, weight, aliases
    )

    latte_results = []

    for i in tqdm.tqdm(range(simplices.shape[0])):
        s = simplices[i].cpu().numpy()
        with TemporaryDirectory(dir=".") as folder:
            integrand_file = latte.INTEGRAND_TEMPLATE
            polytope_file = latte.POLYTOPE_TEMPLATE
            output_file = latte.OUTPUT_TEMPLATE

            # Change the CWD
            original_cwd = os.getcwd()
            os.chdir(folder)

            # Variable ordering is relevant in LattE files
            variables = sorted(integrand.variables.union(polytope.variables))

            # Write integrand and polytope to file
            latte._write_integrand_file(integrand, variables, integrand_file)

            write_polytope_v_file(s, polytope_file)

            cmd = [
                "integrate",
                "--valuation=integrate",
                latte.algorithm,
                "--monomials=" + integrand_file,
                "--vrep",
                polytope_file,
            ]

            with open(output_file, "w") as f:
                return_value = call(cmd, stdout=f, stderr=f)
                if return_value != 0:
                    print(open(output_file).read())
                    """
                    if return_value != 0:
                        msg = "LattE returned with status {}"
                        # LattE returns an exit status != 0 if the polytope is empty.
                        # In the general case this may happen, raising an exception
                        # is not a good idea.
                    """

            # Read back the result and return to the original CWD
            result_latte = latte._read_output_file(output_file)
            os.chdir(original_cwd)
            latte_results.append(result_latte)

    print("hi")