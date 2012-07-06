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

def calc_cond_indep_data(ideal_gas):
    """
    Purpose: Calculate the condition-independent data needed for reaction
    elimination: the molar mass-stoichiometric matrix product

    Arguments:
    ideal_gas (Cantera.Solution): Cantera.Solution object specifying
    a chemical reaction mechanism and the thermodynamic properties of
    its constituent species

    Returns:
    mass_stoich_prod (2-D numpy.ndarray of floats): product of diagonal
    matrix of molar masses and stoichiometry matrix
    """

    molar_mass = ideal_gas.molarMasses()
    stoich_matrix = (ideal_gas.productStoichCoeffs() -
        ideal_gas.reactantStoichCoeffs())
    mass_stoich_prod = numpy.dot(numpy.diag(molar_mass), stoich_matrix)

    return stoich_matrix, mass_stoich_prod

def calc_cond_dep_data(state, ideal_gas):
    """
    Purpose: Calculate the condition-dependent data needed for reaction
    elimination:
    - species mass enthalpies
    - reaction rates
    - mass-based constant pressure heat capacity
    - mass density

    Arguments:
    state (list of floats, or 1-D numpy.ndarray of floats): Reaction
    conditions consisting of temperature and species mass fractions
    (in the order that they are specified in the Cantera mechanism).
    Temperature must be the first element in the system state list
    (or 1-D numpy.ndarray); subsequent elements must be species mass
    fractions, in the order that they are specified in the Cantera
    mechanism.
    ideal_gas (Cantera.Solution): Cantera.Solution object specifying a
    chemical reaction mechanism and the thermodynamic properties of its
    constituent species; uses state of ideal_gas to calculate properties

    Returns:
    rxn_rate (1-D numpy.ndarray of floats): (row) vector of reaction rates
    cp_mass (float): mass-based constant pressure heat capacity
    enthalpy_mass (1-D numpy.ndarray of floats): (row) vector of species
        mass (or specific) enthalpies
    rho (float): mass density

    """
    
    ideal_gas.setTemperature(state[0])
    ideal_gas.setMassFractions(state[1:])
    
    rxn_rate = ideal_gas.netRatesOfProgress()
    cp_mass = ideal_gas.cp_mass()
    enthalpy_mass = (ideal_gas.enthalpies_RT() *
        ideal_gas.temperature() * Cantera.GasConstant)
    rho = ideal_gas.density()
    
    return (rxn_rate, cp_mass, enthalpy_mass, rho)

def error_constraint_data(state, ideal_gas, mass_stoich_prod, atol, rtol):
    """
    Purpose: Calculates all of the coefficients for the error constraints
    in the point-constrained reaction and species elimination integer
    linear programming formulations.

    Arguments:
    state (list of floats, or 1-D numpy.ndarray of floats): Reaction
    conditions consisting of temperature and species mass fractions
    (in the order that they are specified in the Cantera mechanism).
    Temperature must be the first element in the system state list
    (or 1-D numpy.ndarray); subsequent elements must be species mass
    fractions, in the order that they are specified in the Cantera
    mechanism.
    ideal_gas (Cantera.Solution): Cantera.Solution object specifying a
    chemical reaction mechanism and the thermodynamic properties of its
    constituent species; uses state of ideal_gas to calculate properties
    atol (1-D numpy.ndarray of floats): list of absolute tolerances;
    len(atol) == states.shape[1] == ideal_gas.nSpecies() + 1
    rtol (1-D numpy.ndarray of floats): list of relative tolerances;
    len(rtol) == states.shape[1] == ideal_gas.nSpecies() + 1
    mass_stoich_prod (2-D numpy.ndarray of floats): product of diagonal
    matrix of molar masses and stoichiometry matrix

    Returns:
    coeffs_temp (1-D numpy.ndarray of floats): coefficients for constraints
        on error in time derivative of temperature
    coeffs_y (2-D numpy.ndarray of floats): coefficients for constraints on
        on error in time derivatives of species mass fractions
    rhs_temp (float): right-hand side of constraints on error in time
        derivative of temperature
    rhs_y (1-D numpy.ndarray of floats): right-hand side of constraints on
        error in time derivatives of species mass fractions 

    Comments:
    Could refactor this function to use the internal state of ideal_gas,
    but the additional state argument was chosen to make the dependency
    much more explicit.

    """

    (rxn_rate,
     cp_mass,
     enthalpy_mass,
     rho) = calc_cond_dep_data(state, ideal_gas)
    
    coeffs_temp = numpy.dot(enthalpy_mass, numpy.dot(mass_stoich_prod,
        numpy.diag(rxn_rate))) / (rho * cp_mass)
    temp_dot = numpy.dot(coeffs_temp, rxn_rate)
    rhs_temp = atol[0] + rtol[0] * abs(temp_dot)
    
    ydot = numpy.dot(mass_stoich_prod, rxn_rate) / rho
    coeffs_y = numpy.dot(mass_stoich_prod, numpy.diag(rxn_rate)) / rho
    rhs_y = atol[1:] + numpy.dot(abs(ydot), numpy.diag(rtol[1:]))

    return coeffs_temp, coeffs_y, rhs_temp, rhs_y

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
    
    # Convert lists to numpy.ndarrays because the data structure is useful
    # for the operators.
    atol = numpy.asarray(atol)
    rtol = numpy.asarray(rtol)

    # Set up the lists needed for indexing
    rxn_list = range(0, ideal_gas.nReactions())
    rxn_strings = [str(n+1) for n in rxn_list]

    # Instantiate binary variables for integer linear program
    z_var = pulp.LpVariable.dicts('rxn_', rxn_strings, 0, 1, 'Integer')

    # Instantiate integer linear program and objective function
    rxn_elim_ILP = pulp.LpProblem("Reaction Elimination", pulp.LpMinimize)
    rxn_elim_ILP += pulp.lpSum([z_var[s] for s
                                in rxn_strings]), "Number of reactions"

    # Calculate condition-independent data and store
    (stoich_matrix, mass_stoich_prod) = calc_cond_indep_data(ideal_gas)
    ideal_gas.setPressure(Cantera.OneAtm)

    # Add constraints: loop over data points
    for k in range(0, len(states)):

        # Calculate condition-dependent data
        (coeffs_temp, coeffs_y, rhs_temp,
        rhs_y) = error_constraint_data(states[k],
            ideal_gas, mass_stoich_prod, atol, rtol)

        # Add two temperature error constraints (lower, upper bounds)
        rxn_elim_ILP += pulp.lpSum([coeffs_temp[i] *
            (1 - z_var[rxn_strings[i]]) for i in rxn_list]) >= -rhs_t, \
            "Temperature Error Lower Bound for Data Point " + str(k+1)
        rxn_elim_ILP += pulp.lpSum([coeffs_temp[i] *
            (1 - z_var[rxn_strings[i]]) for i in rxn_list]) <= rhs_t, \
            "Temperature Error Upper Bound for Data Point " + str(k+1)

        # Add constraints: Loop over species mass fractions
        for j in range(0, ideal_gas.nSpecies()):
            
            # Add two species mass fraction error constraints (lower, upper
            # bounds)
            rxn_elim_ILP += pulp.lpSum([coeffs_y[j, i] *
                (1 - z_var[rxn_strings[i]]) for i in rxn_list]) >= -rhs_y[j], \
                "Mass Fraction Species " + str(j+1) + \
                " Error Lower Bound for Data Point " + str(k+1)
            rxn_elim_ILP += pulp.lpSum([coeffs_y[j, i] *
                (1 - z_var[rxn_strings[i]]) for i in rxn_list]) <= rhs_y[j], \
                "Mass Fraction Species " + str(j+1) + \
                " Error Upper Bound for Data Point " + str(k+1)

    # Solve integer linear program
    rxn_elim_ILP.solve()

    # Return list of binary variables, solver status
    z = [int(z_var[i].value()) for i in rxn_strings]
    #z = [int(v.value()) for v in rxn_elim_ILP.variables()]
    return z, pulp.LpStatus[rxn_elim_ILP.status]

def reaction_and_species_elim(states, ideal_gas, atol, rtol):
    """
    Purpose: Carries out simultaneous reaction and species
    elimination (Mitsos, et al.,Comb Flame, 2008;
    Mitsos, 2008, unpublished) on the mechanism specified in
    ideal_gas at the conditions specified in states, using the
    absolute tolerances specified in atol, and relative tolerances
    specified in rtol. Mitsos' unpublished formulation is used here,
    which decreases the number of integer variables in the mixed-integer
    linear programming formulation, which decreases its run time compared
    to the original formulation in Combustion and Flame.

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
    w (list of ints, or 1-D numpy.ndarray of ints): binary variables
    indicating which species should be kept, and which should be
    eliminated
    status (str): indicates the LP solver status; is one of "Not
    Solved", "Infeasible", "Unbounded", "Undefined", "Optimal"

    Warnings:
    This function alters the state of ideal_gas. If the state of that
    object prior to calling this function needs to be preserved,
    copy the object.
    
    """

    # Convert lists to numpy.ndarrays because the data structure is useful
    # for the operators.
    atol = numpy.asarray(atol)
    rtol = numpy.asarray(rtol)

    # Set up the lists needed for indexing
    rxn_list = range(0, ideal_gas.nReactions())
    rxn_strings = [str(n+1) for n in rxn_list]

    species_list = range(0, ideal_gas.nSpecies())
    species_strings = [str(n+1) for n in species_list]

    # Instantiate binary variables for integer linear program
    z_var = pulp.LpVariable.dicts('rxn_', rxn_strings, 0, 1, 'Integer')
    w_var = pulp.LpVariable.dicts('species_', species_strings, 0, 1,
                                  'Continuous')

    # Instantiate integer linear program and objective function
    rxn_elim_ILP = pulp.LpProblem("Reaction Elimination", pulp.LpMinimize)
    rxn_elim_ILP += pulp.lpSum([w_var[s] for s
                                in species_strings]), "Number of species"

    # Calculate condition-independent data and store
    (stoich_matrix, mass_stoich_prod) = calc_cond_indep_data(ideal_gas)
    ideal_gas.setPressure(Cantera.OneAtm)

    # Add participation constraints from alternative Mitsos formulation
    for j in range(0, ideal_gas.nSpecies()):
        for i in range(0, ideal_gas.nReactions()):
            if stoich_matrix[j, i] != 0:
                rxn_elim_ILP += \
                    w_var[species_strings[j]] - z_var[rxn_strings[i]] >= 0, \
                    "Participation of species " + str(j+1) + \
                    " and reaction " + str(i+1)
    
    # Add error constraints: loop over data points
    for k in range(0, len(states)):

        # Calculate condition-dependent data
        (coeffs_temp, coeffs_y, rhs_temp,
        rhs_y) = error_constraint_data(states[k],
            ideal_gas, mass_stoich_prod, atol, rtol)

        # Add two temperature error constraints (lower, upper bounds)
        rxn_elim_ILP += pulp.lpSum([coeffs_temp[i] *
            (1 - z_var[rxn_strings[i]]) for i in rxn_list]) >= -rhs_t, \
            "Temperature Error Lower Bound for Data Point " + str(k+1)
        rxn_elim_ILP += pulp.lpSum([coeffs_temp[i] *
            (1 - z_var[rxn_strings[i]]) for i in rxn_list]) <= rhs_t, \
            "Temperature Error Upper Bound for Data Point " + str(k+1)

        # Add constraints: Loop over species mass fractions
        for j in range(0, ideal_gas.nSpecies()):
            
            # Add two species mass fraction error constraints (lower, upper
            # bounds)
            rxn_elim_ILP += pulp.lpSum([coeffs_y[j, i] *
                (1 - z_var[rxn_strings[i]]) for i in rxn_list]) >= -rhs_y[j], \
                "Mass Fraction Species " + str(j+1) + \
                " Error Lower Bound for Data Point " + str(k+1)
            rxn_elim_ILP += pulp.lpSum([coeffs_y[j, i] *
                (1 - z_var[rxn_strings[i]]) for i in rxn_list]) <= rhs_y[j], \
                "Mass Fraction Species " + str(j+1) + \
                " Error Upper Bound for Data Point " + str(k+1)

    # Solve integer linear program
    rxn_elim_ILP.solve()

    # Return list of binary variables, solver status
    z = [int(z_var[i].value()) for i in rxn_strings]
    w = [int(w_var[j].value()) for j in species_strings]
    return z, w, pulp.LpStatus[rxn_elim_ILP.status]
