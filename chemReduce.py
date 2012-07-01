#/usr/bin/env python

# Requirements:
# NumPy
# Cantera
# PuLP

import numpy
import Cantera
import pulp

# Optional:
# Installed LP solvers (Gurobi, CPLEX, CBC, COIN)

def reaction_elim(states, ideal_gas, atol, rtol):
    """
    Purpose: Carries out reaction elimination (Bhattacharjee, et al.,
    Comb Flame, 2003) on the mechanism specified in ideal_gas at the
    conditions specified in states, using the absolute tolerances
    specified in atol, and relative tolerances specified in rtol.

    Arguments:
    states (list of list of floats, or 2-D numpy.ndarray of floats):
    each element of the outer list (or each row of the 2-D
    numpy.ndarray) corresponds to a system state (or condition).
    Conditions consist of temperature and species mass fractions
    (in the order that they are specified in the Cantera mechanism).
    Temperature must be the first element in the system state list
    (or 1-D numpy.ndarray); subsequent elements must be species mass
    fractions, in the order that they are specified in the Cantera
    mechanism.
    ideal_gas (Cantera.Solution): Cantera.Solution object specifying
    a chemical reaction mechanism and the thermodynamic properties of
    its constituent species
    atol (list of floats or 1-D numpy.ndarray of floats): list of
    absolute tolerances; len(atol) == states.shape[1] ==
    ideal_gas.nSpecies() + 1
    rtol (list of floats or 1-D numpy.ndarray of floats): list of
    relative tolerances; len(rtol) == states.shape[1] ==
    ideal_gas.nSpecies() + 1

    Returns:
    z (list of ints, or 1-D numpy.ndarray of ints): binary variables
    indicating which reactions should be kept, and which should be
    eliminated
    status (str): indicates the LP solver status; is one of "Not
    Solved", "Infeasible", "Unbounded", "Undefined", "Optimal"

    Warnings:
    This function alters the state of ideal_gas. If the state of that
    object prior to calling this function needs to be preserved,
    copy the object.
    
    """

    # Set up the lists needed for indexing
    rxn_list = range(0, ideal_gas.nReactions())
    rxn_strings = [str(n) for n in rxn_list]

    # Instantiate binary variables for integer linear program
    z_var = pulp.LpVariable.dicts('rxn_', rxn_strings, 0, 1, 'Integer')

    # Instantiate integer linear program and objective function
    rxn_elim_ILP = pulp.LpProblem("Reaction Elimination", pulp.LpMinimize)
    rxn_elim_ILP += pulp.lpSum([z_var[s] for s
                                in rxn_strings]), "Number of reactions"

    # Calculate condition-independent data and store
    molar_mass = ideal_gas.molarMasses()
    stoich_matrix = ideal_gas.productStoichCoeffs() - \
        ideal_gas.reactantStoichCoeffs()
    mass_stoich_prod = numpy.dot(numpy.diag(molar_mass), stoich_matrix)
    ideal_gas.setPressure(Cantera.OneAtm)

    # Add constraints: loop over data points
    for k in range(0, len(states)):

        # Calculate condition-dependent data
        ideal_gas.setTemperature(sys_state[k][0])
        ideal_gas.setMassFractions(sys_state[k][1:])
        rxn_rate = ideal_gas.netRatesOfProgress()
        cp_mass = ideal_gas.cp_mass()
        enthalpy_mass = ideal_gas.enthalpies_RT() * \
            ideal_gas.temperature() * Cantera.GasConstant
        rho = ideal_gas.density()

        # Add two temperature error constraints (lower, upper bounds)
        coeffs_temp = numpy.dot(enthalpy_mass, mass_stoich_prod) / \
            (rho * cp_mass)
        temp_dot = numpy.dot(coeffs_temp, rxn_rate)
        rxn_elim_ILP += pulp.lpSum([coeffs_temp[i] * rxn_rate[i] *
            (1 - z_var[rxn_strings[i]]) for i in rxn_list]) >= \
            -atol[0] - rtol[0] * abs(temp_dot), \
            "Temperature Error Lower Bound for Data Point " + str(k+1)
        rxn_elim_ILP += pulp.lpSum([coeffs_temp[i] * rxn_rate[i] *
            (1 - z_var[rxn_strings[i]]) for i in rxn_list]) <= \
            atol[0] + rtol[0] * abs(temp_dot), \
            "Temperature Error Upper Bound for Data Point " + str(k+1)

        ydot = numpy.dot(mass_stoich_prod, rxn_rate) / rho
        # Add constraints: Loop over species mass fractions
        for j in range(0, ideal_gas.nSpecies()):
            
            # Add two species mass fraction error constraints (lower, upper
            # bounds)
            rxn_elim_ILP += pulp.lpSum([mass_stoich_prod[j, i] *
                (1 - z_var[rxn_strings[i]]) * rxn_rate[i] / rho
                for i in rxn_list]) >= -atol[j + 1] - rtol[j + 1] * \
                abs(ydot[j]), "Mass Fraction Species " + str(j+1) + \
                " Error Lower Bound for Data Point " + str(k+1)
            rxn_elim_ILP += pulp.lpSum([mass_stoich_prod[j, i] *
                (1 - z_var[rxn_strings[i]]) * rxn_rate[i] / rho
                for i in rxn_list]) <= atol[j + 1] + rtol[j + 1] * \
                abs(ydot[j]), "Mass Fraction Species " + str(j+1) + \
                " Error Upper Bound for Data Point " + str(k+1)

    # Solve integer linear program
    rxn_elim_ILP.solve()

    # Return list of binary variables, solver status
    z = [int(v.value()) for v in rxn_elim_ILP.variables()]
    return z, pulp.LpStatus[rxn_elim_ILP.status]
