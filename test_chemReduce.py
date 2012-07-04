import chemReduce
import Cantera
import unittest
import numpy

class TestCoeffIdentities(unittest.TestCase):
    def setUp(self):
        #methodName='runTest', file_name='gri30.cti', temp=1000,
        #press=Cantera.OneAtm, mass_frac='CH4:.05, O2:.075, N2:.9', atol=1e-6,
        #rtol=1e-6):

        file_name = 'gri30.cti'
        temp = 1000
        press = Cantera.OneAtm
        mass_frac = 'CH4:.05, O2:.075, N2:.9"'
        atol = 1e-6
        rtol = 1e-6

        # Initialize thermodynamic and kinetic data and set state
        self.gas=Cantera.IdealGasMix('gri30.cti')
        self.gas.set(T=temp, P=press, Y=mass_frac)
        self.state = numpy.zeros(self.gas.nSpecies() + 1)
        self.state[0] = self.gas.temperature()
        self.state[1:] = self.gas.massFractions()

        # Calculate condition-independent data and store
        self.mass_stoich_prod = chemReduce.calc_cond_indep_data(self.gas)

        # Calculate condition-dependent data and store
        (self.rxn_rate,
         self.cp_mass,
         self.enthalpy_mass,
         self.rho) = chemReduce.calc_cond_dep_data(self.state, self.gas)

        self.atol = numpy.ones(self.gas.nSpecies() + 1) * atol
        self.rtol = numpy.ones(self.gas.nSpecies() + 1) * rtol

        self.float_tol = 1e-7
        

    def test_row_sums(self):
        """
        Purpose: The sum of the entries in coeffs_temp should equal temp_dot.
        The sum of the entries in each row of coeffs_y should equal
        numpy.asarray([ydot]).transpose().

        Arguments:
        None

        Returns:
        None

        """
        (coeffs_temp,
         coeffs_y,
         rhs_temp,
         rhs_y) = chemReduce.error_constraint_data(self.state,
            self.gas, self.mass_stoich_prod, self.atol, self.rtol)

        # Test identity for temperature
        rhs_temp_test = self.atol[0] + self.rtol[0] * numpy.sum(coeffs_temp)
        self.assertAlmostEqual(rhs_temp_test, rhs_temp, delta=self.float_tol)

        # Test identity for each species
        rhs_y_test = numpy.zeros(self.gas.nSpecies())
        for j in range(0, self.gas.nSpecies()):
            rhs_y_test[j] = (self.atol[j + 1] + self.rtol[j + 1] *
                             numpy.sum(coeffs_y[j,:]))

        self.assertAlmostEqual(numpy.max(abs(rhs_y_test - rhs_y)), 0,
                               delta=self.float_tol)


    def test_col_sums(self):
        """
        Purpose: The sum of over each column in coeffs_y, where each row is
        scaled by enthalpy_mass[j] / cp_mass, should equal coeffs_t.

        Arguments:
        None

        Returns:
        None
        """
        (coeffs_temp,
         coeffs_y,
         _,
         _) = chemReduce.error_constraint_data(self.state,
            self.gas, self.mass_stoich_prod, self.atol, self.rtol)

        row_total = numpy.zeros(self.gas.nReactions())
        for j in range(0, self.gas.nSpecies()):
            row_total += self.enthalpy_mass[j] * coeffs_y[j] / self.cp_mass

        self.assertAlmostEqual(numpy.max(abs(row_total - coeffs_temp)), 0,
                               delta = self.float_tol)


    def test_naive_summation(self):
        """
        Purpose: Calculate the entries of coeffs_temp, coeffs_y, rhs_temp,
        rhs_y using loops instead of vectorizing. Will be slow, but should
        yield same answer.

        Arguments:
        None

        Returns:
        None
        """

        molarMass = self.gas.molarMasses()
        stoichMatrix = (self.gas.productStoichCoeffs() -
            self.gas.reactantStoichCoeffs())

        coeffs_y_loop = numpy.zeros((self.gas.nSpecies(),
                                    self.gas.nReactions()))
        coeffs_temp_loop = numpy.zeros(self.gas.nReactions())

        for i in range(0, self.gas.nReactions()):
            coeffs_temp_loop[i] = numpy.sum(
                [self.enthalpy_mass[j] * molarMass[j] * stoichMatrix[j,i] *
                 self.rxn_rate[i] / (self.cp_mass * self.rho)
                 for j in range(0, self.gas.nSpecies())])
            for j in range(0, self.gas.nSpecies()):
                coeffs_y_loop[j,i] = (molarMass[j] * stoichMatrix[j,i] *
                    self.rxn_rate[i] / self.rho)

        temp_dot = numpy.sum(coeffs_temp_loop)
        y_dot = numpy.sum(coeffs_y_loop, axis=1)

        rhs_temp_loop = self.atol[0] + self.rtol[0] * abs(temp_dot)
        rhs_y_loop = numpy.zeros(self.gas.nSpecies())
        for j in range(0, self.gas.nSpecies()):
            rhs_y_loop[j] = self.atol[j+1] + self.rtol[j+1] * abs(y_dot[j])

        (coeffs_temp,
         coeffs_y,
         rhs_temp,
         rhs_y) = chemReduce.error_constraint_data(self.state,
            self.gas, self.mass_stoich_prod, self.atol, self.rtol)

        self.assertAlmostEqual(rhs_temp, rhs_temp_loop, delta=self.float_tol)

        self.assertAlmostEqual(numpy.max(abs(rhs_y - rhs_y_loop)), 0,
                               delta=self.float_tol)

        self.assertAlmostEqual(numpy.max(abs(coeffs_temp - coeffs_temp)), 0,
                               delta=self.float_tol)

        self.assertAlmostEqual(numpy.max(abs(coeffs_y - coeffs_y_loop)), 0,
                               delta=self.float_tol)

if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(TestCoeffIdentities('test_naive_summation'))
    #suite.addTest(TestCoeffIdentities('test_col_sums'))
    #suite.addTest(TestCoeffIdentities('test_row_sums'))
    #suite.debug()
        
            


        
        

        
        
