import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.optimize import differential_evolution




class five_factor_formula:
    def __init__(
            self,
            N_u235,
            N_u238,
            N_pu239,
            N_gd157,
            N_xe135,
            N_b4c,
            enrichment,
            temperature_fuel,
            moderator_to_fuel_ratio,
            VOID,
            radius_fuel,
            volume_fuel,
            volume_control_rods,
            density_fuel,
            pressure = 7,
            temperature_moderator = 286 + 273,
            fast_fission_factor = 1.06,
            non_leakage_factor = 0.96
    ):

        self.fast_fission_factor = fast_fission_factor
        self.non_leakage_factor = non_leakage_factor

        self.enrichment = enrichment
        self.pu_yield = N_pu239 / N_u235 if N_u235 != 0 and N_pu239 != 0 else 0
        self.mod_to_fuel_ratio = moderator_to_fuel_ratio
        self.temperature_fuel = temperature_fuel
        self.density_fuel = density_fuel

        self.VOID = VOID
        self.pressure = pressure
        self.temperature_moderator = temperature_moderator
        self.steam_table = XSteam(XSteam.UNIT_SYSTEM_BARE)

        self.radius_fuel = radius_fuel
        self.volume_fuel = volume_fuel
        self.volume_mod = self.volume_fuel * moderator_to_fuel_ratio
        self.volume_control_rods = volume_control_rods

        self.N_u235 = N_u235
        self.N_u238 = N_u238
        self.N_pu239 = N_pu239
        self.N_gd157 = N_gd157
        self.N_xe135 = N_xe135
        self.N_b4c = N_b4c

        self.Na = 6.022e23

        self.nu_u235 = 2.4367
        self.nu_pu239 = 2.8789

        self.cs_tot_u235_th = 699.383
        self.cs_tot_u238_th = 11.9943
        self.cs_tot_pu239_th = 1031.11

        self.cs_f_u235_th = 585.472
        self.cs_f_u238_th = 1.68688E-5
        self.cs_f_pu239_th = 751.322

        self.cs_s_u235_th = 15.116
        self.cs_s_u238_th = 9.29983
        self.cs_s_pu239_th = 7.99061

        self.cs_a_u235_th = self.cs_tot_u235_th - self.cs_s_u235_th
        self.cs_a_u238_th = self.cs_tot_u238_th - self.cs_s_u238_th
        self.cs_a_pu239_th = self.cs_tot_pu239_th - self.cs_s_pu239_th

        self.cs_c_u235_th = self.cs_a_u235_th - self.cs_f_u235_th
        self.cs_c_u238_th = self.cs_a_u238_th - self.cs_f_u238_th
        self.cs_c_pu239_th = self.cs_a_pu239_th - self.cs_f_pu239_th


    def thermal_fission_factor(self):
        tff = (self.enrichment * ((1 - self.pu_yield) * self.nu_u235 * self.cs_f_u235_th
                                  + self.pu_yield * self.nu_pu239 * self.cs_f_pu239_th) /
               (self.enrichment * ((1 - self.pu_yield) * self.cs_a_u235_th+ self.pu_yield * self.cs_a_pu239_th) +
                (1 - self.enrichment) * self.cs_c_u238_th))
        return tff


    def a_cross_section_moderator(self):
        # Calculates the macroscopic absorption cross section for water
        cs_a_moderator_th = 2 * (30.6083 - 30.1468) + (3.852000 - 3.851810)

        density_water = self.steam_table.rhoL_p(self.pressure) / 1000
        density_steam = self.steam_table.rho_pt(self.pressure, self.temperature_moderator) / 1000
        density_moderator = self.VOID * density_steam + (1 - self.VOID) * density_water

        molar_mass = 18.015

        M_cs_a_moderator = cs_a_moderator_th * (self.Na * density_moderator / molar_mass) * 1E-24

        return M_cs_a_moderator


    def a_cross_section_fuel(self):
        # Calculates the macroscopic absorption cross section for the fuel
        M_cs_a_u235_th = self.N_u235 * self.cs_a_u235_th * 1E-24
        M_cs_a_u238_th = self.N_u238 * self.cs_a_u238_th * 1E-24
        M_cs_a_pu239_th = self.N_pu239 * self.cs_a_pu239_th * 1E-24

        cs_a_o_th = 3.852000 - 3.851810
        M_cs_a_o_th = 2 * (self.N_u238 + self.N_u235 + self.N_pu239) * cs_a_o_th * 1E-24

        return M_cs_a_pu239_th + M_cs_a_u238_th + M_cs_a_u235_th + M_cs_a_o_th


    def a_cross_section_gd157(self):
        # Calculates the macroscopic absorption cross section for Gd157
        cs_a_gd157_th = 254554.0 - 1000.16
        M_cs_a_gd157_th = self.N_gd157 * cs_a_gd157_th * 1E-24

        return M_cs_a_gd157_th


    def a_cross_section_control_rod(self):
        cs_a_b10_th = 3.845241E3- 2.175000
        cs_a_c6_th = 4.743100 - 4.739240
        M_cs_a_b10_th = self.N_b4c * (4 * cs_a_b10_th + cs_a_c6_th) * 1E-24

        return M_cs_a_b10_th


    def a_cross_section_xe135(self):
        # Calculates the macroscopic absorption cross section for Gd157
        cs_a_xe135_th = 2968600.0 - 299670.0
        M_cs_a_xe135 = self.N_xe135 * cs_a_xe135_th * 1E-24

        return M_cs_a_xe135


    def thermal_utilization_factor(self,vol_fuel, vol_mod):
        M_cs_a_moderator = self.a_cross_section_moderator()
        M_cs_a_fuel = self.a_cross_section_fuel()
        M_cs_a_gd157 = self.a_cross_section_gd157()
        M_cs_a_b4c = self.a_cross_section_control_rod()
        M_cs_a_xe135 = self.a_cross_section_xe135()

        mod_to_fuel_ratio = vol_mod / vol_fuel
        cr_to_fuel_ratio = self.volume_control_rods / vol_fuel

        tuf = M_cs_a_fuel / (M_cs_a_fuel + M_cs_a_gd157 + M_cs_a_xe135 + mod_to_fuel_ratio * M_cs_a_moderator +
                             cr_to_fuel_ratio * M_cs_a_b4c)

        return tuf


    def s_cross_section_moderator(self):
        # Calculates the macroscopic scattering cross section for water
        cs_s_moderator_th = 2 * 30.1468 + 3.9759

        density_water = self.steam_table.rhoL_p(self.pressure) / 1000
        density_steam = self.steam_table.rhoV_p(self.pressure) / 1000
        density_moderator = self.VOID * density_steam + (1 - self.VOID) * density_water

        molar_mass = 18.015

        M_cs_s_moderator = cs_s_moderator_th * (self.Na * density_moderator / molar_mass) * 1E-24

        return M_cs_s_moderator


    def resonance_escape_probability(self, vol_fuel, vol_mod):
        mod_to_fuel_ratio = vol_mod / vol_fuel

        B1 = 6.1E-3 + 0.94E-2 / (self.radius_fuel * self.density_fuel)
        cs_a_i_300k = 3 + 39.6 / np.sqrt(self.radius_fuel * self.density_fuel)
        cs_a_i_T = cs_a_i_300k * (1 + B1 * (np.sqrt(self.temperature_fuel) - np.sqrt(300))) * 1E-24
        M_cs_s_mod = self.s_cross_section_moderator()
        xi = 0.9191542      # avg log energy loss

        rep = np.exp(-(1 - self.enrichment) * self.N_u238 * cs_a_i_T / (mod_to_fuel_ratio * M_cs_s_mod * xi))

        return rep


    def calculate(self):
        tff = self.thermal_fission_factor()
        tuf = self.thermal_utilization_factor(vol_fuel=self.volume_fuel, vol_mod=self.volume_mod)
        rep = self.resonance_escape_probability(vol_fuel=self.volume_fuel, vol_mod=self.volume_mod)
        k = tff * tuf * rep * self.non_leakage_factor * self.fast_fission_factor

        return k


    def calculate_tuf_rep_function(self, display_results = False):
        """
        Function plots the p*f as a function of Vm/Vf if display_results = True.

        Returns True if we have a negative coefficient (i.e. if the evaluated data point
        is situated to the left of the p*f curves maximum value).
        """

        mod_fuel_ratio_vec = np.linspace(0.01,10,5000)
        vol_mod_vec = mod_fuel_ratio_vec * self.volume_fuel

        p = self.resonance_escape_probability(vol_fuel=self.volume_fuel, vol_mod=vol_mod_vec)
        f = self.thermal_utilization_factor(vol_fuel=self.volume_fuel, vol_mod=vol_mod_vec)
        pf = np.multiply(p, f)

        max_index = np.argmax(pf)
        pf_max = pf[max_index]
        mod_fuel_max = mod_fuel_ratio_vec[max_index]

        p_eval = self.resonance_escape_probability(vol_fuel=self.volume_fuel, vol_mod=self.volume_mod)
        f_eval = self.thermal_utilization_factor(vol_fuel=self.volume_fuel, vol_mod=self.volume_mod)
        pf_eval = p_eval*f_eval

        if display_results:
            print(f'\nMaximum value:      p*f = {pf_max},    Vm/Vf = {mod_fuel_max}')
            print(f'Evaluated value:    p*f = {pf_eval},    Vm/Vf = {self.volume_mod / self.volume_fuel}')
            if (self.volume_mod / self.volume_fuel) < mod_fuel_max:
                print('We have a negative VOID coefficient\n')
            else:
                print('WARNING: We have a positive VOID coefficient\n')


            plt.figure()
            plt.xlabel(r'$V_m/V_f$')
            plt.grid()
            plt.plot(mod_fuel_ratio_vec, p, label=r'$p$')
            plt.plot(mod_fuel_ratio_vec, f, label=r'$f$')
            plt.plot(mod_fuel_ratio_vec, pf, label=r'$p \cdot f$')
            plt.vlines(x=mod_fuel_max, ymin=0, ymax=1, ls='dashed',
                       label=r'$\text{maximum } p \cdot f$')
            plt.plot(self.volume_mod / self.volume_fuel, pf_eval, marker='o', ls=None, label='current value')
            plt.legend()
            plt.show()

        return True if (self.volume_mod / self.volume_fuel) < mod_fuel_max else False





class reactor:
    def __init__(
            self,
            m_fuel_tot = 163,
            temperature_fuel = 600,
            temperature_moderator = 286,
            pressure = 7,
            thermal_power = 4500,
            cycle_length = 18,
            radius_fuel = 0.35,
            length_fuel = 379,
            num_control_rods = 269,
            num_fuel_assemblies = 1132,
            fast_fission_factor=1.06,
            non_leakage_factor=0.96
    ):

        """
        m_fuel_tot: [tons],
        temoerature_fuel: [deg C],
        temperature_moderator: [deg C],
        pressure [MPa],
        thermal_power: [MW],
        cycle_length: [months],
        radius_fuel: [cm],
        length_fuel: [cm],
        num_control_rods,
        num_fuel_assemblies,
        fast_fiction_factor,
        non_leakage_factor
        """

        self.fast_fission_factor = fast_fission_factor
        self.non_leakage_factor = non_leakage_factor

        self.temperature_fuel = temperature_fuel + 273
        self.temperature_moderator = temperature_moderator + 273
        self.pressure = pressure

        self.cycle_length_seconds = cycle_length / 12 * 365 * 24 * 60 * 60

        self.radius_fuel = radius_fuel
        self.length_fuel = length_fuel
        self.num_fuel_assemblies = num_fuel_assemblies
        self.num_control_rods = num_control_rods

        self.volume_fuel = self.calculate_volume_fuel()
        self.m_fuel_tot = m_fuel_tot * 1E6
        self.density_fuel = m_fuel_tot / self.volume_fuel * 1E6

        self.thermal_power = thermal_power * 1E6     # Power of reactor [W]
        self.E_fission = 200 * 1E6 * 1.602E-19      # Energy per fission [J]
        self.fission_rate = self.thermal_power/self.E_fission

        self.Na = 6.022e23

        self.nu_u235 = 2.4367
        self.nu_pu239 = 2.8789

        self.cs_tot_u235_th = 699.383
        self.cs_tot_u238_th = 11.9943
        self.cs_tot_pu239_th = 1031.11

        self.cs_f_u235_th = 585.472
        self.cs_f_u238_th = 1.68688E-5
        self.cs_f_pu239_th = 751.322

        self.cs_s_u235_th = 15.116
        self.cs_s_u238_th = 9.29983
        self.cs_s_pu239_th = 7.99061

        self.cs_a_u235_th = self.cs_tot_u235_th - self.cs_s_u235_th
        self.cs_a_u238_th = self.cs_tot_u238_th - self.cs_s_u238_th
        self.cs_a_pu239_th = self.cs_tot_pu239_th - self.cs_s_pu239_th

        self.cs_c_u235_th = self.cs_a_u235_th - self.cs_f_u235_th
        self.cs_c_u238_th = self.cs_a_u238_th - self.cs_f_u238_th
        self.cs_c_pu239_th = self.cs_a_pu239_th - self.cs_f_pu239_th


    def calculate_volume_fuel(self):
        volume_fuel_rod = np.pi * self.radius_fuel ** 2 * self.num_fuel_assemblies * self.length_fuel * 92
        return volume_fuel_rod


    def calculate_volume_control_rods(self, control_rod_insertion_percentage):
        # Approximating the dimensions of the control rod, by approximating the fuel rod spacing
        width = 2 * self.radius_fuel
        length = 24 * self.radius_fuel
        volume_control_rods = (self.num_control_rods * 4 * control_rod_insertion_percentage * self.length_fuel *
                               width * length)

        return volume_control_rods


    def N_nuclides(self, m_fuel, m_ba, enrichment, pu_yield):
        # Fuel rods
        molar_mass_u235 = 235.04
        molar_mass_u238 = 238.03
        molar_mass_pu239 = 239.05
        molar_mass_o = 15.999
        molar_mass_gd157 = 157.25

        molar_mass_fuel = (2 * molar_mass_o + (1 - enrichment) * molar_mass_u238 + enrichment *
                           (pu_yield * molar_mass_pu239 + (1 - pu_yield) * molar_mass_u235))

        molar_mass_ba = 2 * molar_mass_gd157 + 3 * molar_mass_o

        N_u235 = enrichment * (1 - pu_yield) * self.Na * m_fuel / molar_mass_fuel
        N_u238 = (1 - enrichment) * self.Na * m_fuel / molar_mass_fuel
        N_pu239 = enrichment * pu_yield * self.Na * m_fuel / molar_mass_fuel
        N_gd157 = 2 * self.Na * m_ba / molar_mass_ba

        N_u235 /= self.volume_fuel
        N_u238 /= self.volume_fuel
        N_pu239 /= self.volume_fuel
        N_gd157 /= self.volume_fuel

        return N_u235, N_u238, N_pu239, N_gd157


    def N_cr(self):
        # assumes 5% of cr volume is actually made of B4C
        density_b4c = 2.50
        molar_mass_b4c = 55.255

        N_b4c = 0.05 * self.Na * density_b4c / molar_mass_b4c

        return N_b4c


    def neutron_flux(self, N_u235, N_pu239):
        if N_u235 == 0 and N_pu239 == 0:
            return 0
        else:
            phi = self.fission_rate / (self.cs_f_u235_th * N_u235 + self.cs_f_pu239_th * N_pu239)
            return phi


    def N_xe(self, N_u235, N_pu239):
        flux = self.neutron_flux(N_u235, N_pu239)

        cs_a_xe135_th = 2968600.0 - 299670.0
        M_cs_f_fuel = self.cs_f_pu239_th * N_pu239 + self.cs_f_u235_th * N_u235 * 1E-24

        fission_yield_xe = 0.0025
        fission_yield_i = 0.064

        half_life_xe = 9.14 * 60 * 60
        decay_constant_xe = np.log(2) / half_life_xe

        N_xe135 = M_cs_f_fuel * flux * (fission_yield_xe + fission_yield_i) / (decay_constant_xe + flux * cs_a_xe135_th)

        return N_xe135


    def conversion_ratio(self, fff):
        N_u235 = getattr(fff, 'N_u235')
        N_u238 = getattr(fff, 'N_u238')
        mod_to_fuel_ratio = getattr(fff, 'mod_to_fuel_ratio')

        M_cs_f_u235 = N_u235 * self.cs_f_u235_th * 1E-24
        M_cs_a_u235 = N_u235 * self.cs_a_u235_th * 1E-24
        M_cs_a_u238 = N_u238 * self.cs_a_u238_th * 1E-24

        volume_moderator = self.volume_fuel * mod_to_fuel_ratio

        p = fff.resonance_escape_probability(vol_fuel = self.volume_fuel, vol_mod = volume_moderator)

        c = (M_cs_a_u238 / M_cs_a_u235 + M_cs_f_u235 / M_cs_a_u235 *
             self.fast_fission_factor * self.nu_u235 * self.non_leakage_factor * (1 - p))

        return c


    def N_used_fissile_nuclei(self, fff):
        c = self.conversion_ratio(fff)
        delta_N_fissile = (1 - c) * self.fission_rate * self.cycle_length_seconds * self.cs_a_u235_th/self.cs_f_u235_th
        delta_N_fissile /= self.volume_fuel

        return delta_N_fissile


    def calculate(
            self,
            mod_to_fuel_ratio = 2.5,
            ba_mass_fraction_start = 0.01,
            enrichment_start = 0.025,
            VOID_start = 0.4,
            control_rod_insertion_start = 0.0001,
            VOID_end = 0.1,
            control_rod_insertion_end = 0.01,
            pu_yield_end = 0.4,
            display_results = True
    ):

        vol_control_rods_start = self.calculate_volume_control_rods(control_rod_insertion_start)
        m_fuel_start = self.m_fuel_tot * (1 - ba_mass_fraction_start)
        m_ba_start = self.m_fuel_tot * ba_mass_fraction_start

        N_u235_start, N_u238_start, N_pu239_start, N_gd157_start = self.N_nuclides(m_fuel = m_fuel_start,
                                                                                   m_ba = m_ba_start,
                                                                                   enrichment = enrichment_start,
                                                                                   pu_yield = 0)

        N_xe135_start = self.N_xe(N_u235_start, N_pu239_start)
        N_b4c = self.N_cr()

        fff_start = five_factor_formula(N_u235_start, N_u238_start, N_pu239_start, N_gd157_start, N_xe135_start,
                                        N_b4c, enrichment_start, self.temperature_fuel, mod_to_fuel_ratio,
                                        VOID_start, self.radius_fuel, self.volume_fuel, vol_control_rods_start,
                                        self.density_fuel, self.pressure, self.temperature_moderator)

        
        k_start = fff_start.calculate()
        negative_coeff_start = fff_start.calculate_tuf_rep_function(display_results = False)

        delta_N_fissile = self.N_used_fissile_nuclei(fff_start)

        if N_u235_start - delta_N_fissile < 0:
            N_u235_end = 0
            N_pu239_end = 0
        else:
            N_u235_end = (1 - pu_yield_end) * (N_u235_start - delta_N_fissile)
            N_pu239_end = pu_yield_end * (N_u235_start - delta_N_fissile)

        enrichment_end = (N_u235_end + N_pu239_end) / N_u238_start

        N_xe135_end = self.N_xe(N_u235_end, N_pu239_end)
        N_gd157_end = 0

        vol_control_rods_end = self.calculate_volume_control_rods(control_rod_insertion_end)

        fff_end = five_factor_formula(N_u235_end, N_u238_start, N_pu239_end, N_gd157_end, N_xe135_end, N_b4c,
                                      enrichment_end, self.temperature_fuel, mod_to_fuel_ratio, VOID_end,
                                      self.radius_fuel, self.volume_fuel, vol_control_rods_end, self.density_fuel,
                                      self.pressure, self.temperature_moderator)

        k_end = fff_end.calculate()
        negative_coeff_end = fff_end.calculate_tuf_rep_function(display_results = False)

        if display_results:
            print('\nSTART:')
            print(f'e_start = {enrichment_start}')
            print(f'k_start = {k_start}')
            _ = fff_start.calculate_tuf_rep_function(display_results = True)

            c = self.conversion_ratio(fff_start)
            print(f'c = {c}')

            print(f'\nEND:')
            print(f'e_end = {enrichment_end}')
            print(f'k_end = {k_end}')
            _ = fff_end.calculate_tuf_rep_function(display_results = True)

        return (k_start, k_end), (negative_coeff_start, negative_coeff_end)




# Objective function for optimization
def objective_function(x, r, mod_to_fuel_ratio, enrichment, pu_yield_end):
    # Hyperparameters
    p = 6
    t = 0.0001

    ba_mass_fraction, VOID_start, cr_insertion_start, VOID_end, cr_insertion_end = x

    k_vec, coeff_vec = r.calculate(
        mod_to_fuel_ratio = mod_to_fuel_ratio,
        ba_mass_fraction_start = ba_mass_fraction,
        enrichment_start = enrichment,
        VOID_start = VOID_start,
        control_rod_insertion_start = cr_insertion_start,
        VOID_end = VOID_end,
        control_rod_insertion_end = cr_insertion_end,
        pu_yield_end = pu_yield_end,
        display_results = False
    )

    if all(coeff_vec):
        err_start = abs(k_vec[0] - 1)
        err_end = abs(k_vec[1] - 1)
        Lp_norm = (err_start**p + err_end**p)**(1/p)

        loss = Lp_norm + t*(cr_insertion_start + cr_insertion_end)

    else:
        loss = 1E9

    return loss





def main(random = False):
    mod_to_fuel_ratio = 2.5
    enrichment = 0.025
    pu_yield_end = 0.4

    r = reactor()

    bounds = [(0, 0.1), (0, 1), (0, 1), (0, 1), (0, 1)]
    optimized_values = differential_evolution(
        objective_function,
        bounds = bounds,
        args = (r, mod_to_fuel_ratio, enrichment, pu_yield_end),
        maxiter = 100,
        seed = None if random else 100
    )

    ba_mass_fraction = optimized_values.x[0]
    VOID_start = optimized_values.x[1]
    cr_start = optimized_values.x[2]
    VOID_end = optimized_values.x[3]
    cr_end = optimized_values.x[4]
    print('OPTIMIZATION RESULTS')
    print(f'Mass fraction of burnable absorbers:    {ba_mass_fraction}')
    print('             Start        End')
    print(f'CR:         {100*cr_start:.2f}%         {100*cr_end:.2f}%')
    print(f'VOID:       {100*VOID_start:.2f}%       {100*VOID_end:.2f}%\n\n')

    _,_ = r.calculate(
        mod_to_fuel_ratio = mod_to_fuel_ratio,
        ba_mass_fraction_start = ba_mass_fraction,
        enrichment_start = enrichment,
        VOID_start = VOID_start,
        control_rod_insertion_start = cr_start,
        VOID_end = VOID_end,
        control_rod_insertion_end = cr_end,
        pu_yield_end = pu_yield_end,
        display_results = True
    )





if __name__ == '__main__':
    main()






