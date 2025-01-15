######################################################
#  Main file running the code of:
##################
#  A toolkit for solving overlapping generations models with
#  family-linked bequest and intergenerational skill transmission
##################

####
# Structure:
#   Part 1: Preambles
#   Part 2: Define parameters
#   Part 3: Solve full model
#       - Baseline calibration with estimated parameters
#   Part 4: Simulation of the full model
#   Part 5: Estimation (outcommented)
#   Part 6: Plots and tables of full model
#   Part 7: Simulation of the model with random skill transmission
#   Part 8: Solve and simulate model with no bequest motive

########################################
# Part 1: Perambles

import time

import matplotlib.pyplot as plt  # needed if Part 5 Estimation uncommented
import model as model
import numpy as np
import scipy.optimize as optimize  # needed if Part 5 Estimation uncommented
import scipy.stats as stats
import tools as tools

np.set_printoptions(suppress=True)

# define your directory for figures and tex-files for tables
save_to = "C:/Users/Stohler/Documents/GitHub/OLGLinkedBequest/figtabs/"

########################################
# Part 2: Define parameters

par = dict()
###
# a) fixed parameters
par["T"] = 6
par["r"] = 4  # Number of periods in the workforce
par["ch"] = 4  # Age when children are 1
par["o"] = 1  # scale parameter, set to 1 and thus not mentioned in the paper

###
# b) adjustable economic parameters
par["β"] = 0.85  # patience
par["R"] = 1 / par["β"]  # exogenous interest rate
par["w"] = 1  # wage rate
par["ρ"] = 2  # Inverse elasticity of intertemporal substitution of consumption
par["γ"] = 2  # Inverse elasticity of intertemporal substitution of warm-glow bequest
par["a̲"] = 12.49087704  # 12.38 # bequest shifter
par["κ"] = 4.68242024  # 4.74 # strength of bequest motive
par["σ_eps"] = np.sqrt(
    0.3
)  # std of epsilon / normal shock in the AR1 // variance = 0.3
par["α"] = 0.85  # persistence in productivity shock / AR1 paramenter
par["σ_z"] = np.sqrt(
    par["σ_eps"] ** 2 / (1 - par["α"] ** 2)
)  # unconditional std of productivity
par["S"] = 7  # number of Gauss-Hermite nodes
par["ψ"] = np.concatenate(
    [
        np.ones([par["r"]]),
        np.array([0.83, 0.58]) * np.ones([par["T"] - par["r"]]),
        np.zeros([par["ch"]]),
    ]
)  # Mortality pattern course SSA
par["l"] = np.concatenate(
    [
        np.array([48.6, 74.3, 77.8, 63.6])
        / np.mean(np.array([48.6, 74.3, 77.8, 63.6])),
        np.zeros([par["T"] - par["r"] + 1]),
    ]
)  # Age dependend-prod pattern (CBO)
quintiles_income = np.array(
    [350, 480, 659, 917, 2149]
)  # income by Quintiles USA 2016 Source: CBO, The distribution of household income, 2016
par["q"] = np.repeat(0.2, 5)  # quintile populaiton shares
par["H"] = (
    quintiles_income / np.sum(quintiles_income) / par["q"]
)  # human captial / permanent skill/income heterogeneity
par["P_markov"] = np.array(
    [
        [0.337, 0.242, 0.178, 0.134, 0.109],
        [0.28, 0.242, 0.198, 0.16, 0.12],
        [0.184, 0.217, 0.221, 0.208, 0.17],
        [0.124, 0.176, 0.22, 0.244, 0.236],
        [0.075, 0.123, 0.183, 0.254, 0.365],
    ]
)


###
# c) adjustable computational parameters
par["M_max"] = (
    par["H"][-1]
    * par["w"]
    * par["r"]
    * np.mean(par["l"][0 : par["r"]])
    * stats.lognorm.ppf(0.99, par["σ_z"])
)  # Endogenous max of cash-on-hand grid
par["gridsize_m"] = 20  # Cash-on-hand gridsize in the stochastic part of the model
par["gridsize_deterministic_m"] = (
    20  # Cash-on-hand gridsize in the deterministic part of the model
)
par["gridsize_z"] = 5  # Gridsize of labour productivity process
par["𝒢_m_det"] = tools.nonlinspace(
    10e-6, par["M_max"], par["gridsize_deterministic_m"] - 1, 1
)  # Grid of cash-on-hand in the final period
par["𝒢_a_det"] = tools.nonlinspace(
    10e-6, par["M_max"], par["gridsize_deterministic_m"] - 1, 1
)  # for z we choose to only extrapolate for 1%  // np.sqrt(par['σ']^2/(1-par['α']^2) = unconditional std of z
par["𝒢_z"] = tools.nonlinspace(
    -2.576 * par["σ_z"], 2.576 * par["σ_z"], par["gridsize_z"], 1
)  # 99%-confidenceband around


########################################
# Part 3: Solve the model with heavily vectorized code

start = time.time()

# Unpacking Parameters
β = par["β"]
γ = par["γ"]
a̲ = par["a̲"]
κ = par["κ"]
σ_z = par["σ_z"]
ψ = par["ψ"]
T = par["T"]
ch = par["ch"]
o = par["o"]
R = par["R"]
ρ = par["ρ"]
S = par["S"]
l = par["l"]
H = par["H"]
w = par["w"]
α = par["α"]
gridsize_z = par["gridsize_z"]
gridsize_m = par["gridsize_m"]
gridsize_deterministic_m = par["gridsize_deterministic_m"]
𝒢_m_det = par["𝒢_m_det"]
𝒢_a_det = par["𝒢_a_det"]
𝒢_z = par["𝒢_z"]
H = par["H"]

# Preallocating
H_Cstar = []  # This list will consist of Cstar for different levels of h
H_𝒢_MEn = []  # This list will consist of Cash-on-hand grids for different levels of h
Cstar_det = np.zeros(
    [T, gridsize_deterministic_m]
)  # Cstar for deterministic periods // Note, dim1=6, despite only 3 rows filled
𝒢_M_det = np.zeros(
    [T, gridsize_deterministic_m]
)  # Endogenous grid for cash-on-hand in deterministic periods // Note, dim1=6, despite only 3 rows filled

# Load Gauss-Hermite weights and nodes
x, wi = tools.gauss_hermite(S)
ω = wi / np.sqrt(np.pi)  # Scaling the weights

###################################
# Part 1: Deterministic problem
# - Parents are dead, and in the next periods agents are retired
# - There are no more income and bequest shocks => thus 'deterministic'
# - The only risk left is mortality risk

# A) Solve the final period (independent of skill levels)
cT_vec = model.solveT(
    par
)  # Solve the ultimate period for optimal consumption // Over the exogenous grid par['𝒢_m_det']
𝒢_M_det[T - 1, :] = np.append(0, 𝒢_m_det)  # Add a zero as first input (Savings>0)
Cstar_det[T - 1, :] = np.append(
    0, cT_vec
)  # With zero cash-on-hand, you get zero consumption

print("\nStart of solving the model")
print("0.0%")
# B) All the other deterministic periods
for t in range(T - 2, T - ch, -1):
    ################################################################################################################
    # Extracting and vectorizing computation
    m_plus = R * 𝒢_a_det  # Vector of cash-on-hand values for tomorrow

    # Vectorized consumption interpolation
    # consumption tomorrow
    C_plus = tools.interp_linear_1d(𝒢_M_det[t + 1, :], Cstar_det[t + 1, :], m_plus)

    # Vectorized computation for Cstar_det and 𝒢_M_det
    # opt. consumption today / Euler
    Cstar_det[t, 1:] = o * (
        R * β * ψ[t + 1] * (C_plus / o) ** (-ρ)
        + (1 - ψ[t + 1]) * κ * ((𝒢_a_det + a̲) / o) ** (-γ)
    ) ** (-1 / ρ)

    # endogenous cash-on-hand grid of the deterministic periods < T
    𝒢_M_det[t, 1:] = 𝒢_a_det + Cstar_det[t, 1:]

###################################
# Part 2: Stochastic, non-deterministic problem
# - Parents could still be alive and agents will be in the work force in the next period
# - Agents face income shocks and potentially bequest shocks, no mortality risk

############################################################################################################################
#  0 ) Loop over skill levels
for ih, h in enumerate(H):
    Cstar_sto = np.zeros(
        [T, gridsize_z, gridsize_m, 2, gridsize_deterministic_m]
    )  # storage for Cstar for a specific skill level h, later appended to list H_Cstar
    𝒢_MEn_sto = np.zeros(
        [T, gridsize_z, gridsize_m, 2, gridsize_deterministic_m]
    )  # storage for cash-on-hand grid for a specific skill level h, later appended to list H_𝒢_MEn

    ############################################################################################################################
    #  1 ) Loop over last period productivity level
    for iz, z in enumerate(𝒢_z):
        # Gauss-Hermite draws
        ez = np.exp(
            σ_z * np.sqrt(2) * x + α * z
        )  # e^{z_t} = np.exp(σ*np.sqrt(2)*x + rho*z_{t-1} ) see slides
        ez_shock = ez.flatten()  # S shock nodes

        ############################################################################################################################
        # 2 ) Loop backwards over time periods
        for t in range(T - ch, -1, -1):
            # with T = 6 and ch=4 we have t=2,1,0
            A_max_h = (
                h * w * np.sum(l[0 : (t + 1)]) * stats.lognorm.ppf(0.99, σ_z)
            )  # the t times 99 percentil would have this amount of cash-on-hand to potentionally be saved
            𝒢_a_h = np.linspace(
                10e-6, A_max_h, gridsize_m - 1
            )  # constant gridsize even though M_max_a_h changes with h and z

            ########################################################################################################################
            # 3 ) Loop over parents cash-on-hand
            for imp, mp in enumerate(
                𝒢_M_det[t + ch - 1, :]
            ):  # parents are in the deterministic periods of life
                Cparents = Cstar_det[
                    t + ch - 1, imp
                ]  # Cstar[T-1] and Cstar[T-2] is independent of grandparents wealth, because they have died
                ap = mp - Cparents  # savings
                mp_plus = np.repeat(R * ap, S)  # cash-on-hand tomorrow
                b = mp_plus  # potential bequest
                ####################################################################################################################
                # 4 ) Parents Dead or Alive?
                for φ in np.array([0, 1]):
                    ################################################################################################################
                    # 5 ) Loop over child savings
                    for ia, a in enumerate(𝒢_a_h):
                        m_plus = (
                            w * h * l[t + 1] * ez_shock
                            + R * a
                            + b
                            * np.array(
                                [
                                    np.array([0, 1]),
                                ]
                                * S
                            ).T
                        )  # cash-on-hand tomorrow dim=[2,S] first row without bequest, second row with bequest
                        if (
                            φ == 0
                        ):  # parents haven't died yet and we might get bequest tomorrow
                            if (
                                t == T - ch
                            ):  # t=2, Age=3 // we know our parents (if they are still alive) will die in t==3
                                C_plus = tools.interp_linear_1d(
                                    𝒢_M_det[t + 1, :],
                                    Cstar_det[t + 1, :],
                                    m_plus[1,],
                                )  # consumption tomorrow given cash-on-hand tomorrow 'm_plus'
                                Cstar_sto[t, iz, ia + 1, φ, imp] = (
                                    R * β * ψ[t + 1] * (ω @ C_plus ** (-ρ))
                                ) ** (
                                    -1 / ρ
                                )  # optimal consumption in stochastic period for given m,z,mp
                            else:  # t=0,1, Age=1,2
                                # parents wealth tomorrow 'mp_plus' can jump off the endogous grid 𝒢_M_det[t+ch,:]
                                # This is why we are doing a binary search on the grid and then take a weighted average of corresponding
                                # optimal consumption tomorrow if parents survive
                                imp_plus_min = tools.binary_search(
                                    0,
                                    gridsize_deterministic_m,
                                    𝒢_M_det[t + ch, :],
                                    mp_plus[0],
                                )  # between which two gridpoints does parents wealth fall?
                                imp_plus_max = imp_plus_min + 1

                                # caluclation of consumption tomorrow given cash-on-hand tomorrow 'm_plus'if parents survive
                                C_plus0_min = tools.interp_linear_1d(
                                    𝒢_MEn_sto[t + 1, iz, :, 0, imp_plus_min],
                                    Cstar_sto[t + 1, iz, :, 0, imp_plus_min],
                                    m_plus[0,],
                                )  # If parent survives
                                C_plus0_max = tools.interp_linear_1d(
                                    𝒢_MEn_sto[t + 1, iz, :, 0, imp_plus_max],
                                    Cstar_sto[t + 1, iz, :, 0, imp_plus_max],
                                    m_plus[0,],
                                )  # If parent survives
                                weight = (
                                    mp_plus[0] - 𝒢_M_det[t + ch, imp_plus_min]
                                ) / (
                                    𝒢_M_det[t + ch, imp_plus_max]
                                    - 𝒢_M_det[t + ch, imp_plus_min]
                                )
                                C_plus0 = (
                                    1 - weight
                                ) * C_plus0_min + weight * C_plus0_max

                                # caluclation of consumption tomorrow given cash-on-hand tomorrow 'm_plus'if parents have died
                                C_plus1 = tools.interp_linear_1d(
                                    𝒢_MEn_sto[t + 1, iz, :, 1, 0],
                                    Cstar_sto[t + 1, iz, :, 1, 0],
                                    m_plus[1,],
                                )  # if parent has just died

                                # calulation of consumption today given expected consupmtion tomorrow
                                exp_uprime = ψ[t + ch] * ω @ C_plus0 ** (-ρ) + (
                                    1 - ψ[t + ch]
                                ) * ω @ C_plus1 ** (-ρ)
                                Cstar_sto[t, iz, ia + 1, φ, imp] = (
                                    R * β * ψ[t + 1] * exp_uprime
                                ) ** (
                                    -1 / ρ
                                )  ## optimal consumption today in stochastic period for given m,z,mp
                        else:  # φ==1 // We have already received bequest because parents are dead
                            if (
                                t == T - ch
                            ):  # t=2, Age=3 // we have to use the deterministic grid
                                C_plus = tools.interp_linear_1d(
                                    𝒢_M_det[t + 1, :],
                                    Cstar_det[t + 1, :],
                                    m_plus[0,],
                                )  # consumption tomorrow given cash-on-hand
                            else:  # t=0,1, Age=1,2 // we have to use the endogenous grid
                                C_plus = tools.interp_linear_1d(
                                    𝒢_MEn_sto[t + 1, iz, :, 1, 0],
                                    Cstar_sto[t + 1, iz, :, 1, 0],
                                    m_plus[0,],
                                )  # consumption tomorrow given cash-on-hand
                            Cstar_sto[t, iz, ia + 1, φ, imp] = (
                                R * β * ψ[t + 1] * ω @ (C_plus ** (-ρ))
                            ) ** (-1 / ρ)
                        𝒢_MEn_sto[t, iz, ia + 1, φ, imp] = (
                            a + Cstar_sto[t, iz, ia + 1, φ, imp]
                        )  # endogenous grid

    H_Cstar.append(Cstar_sto)
    H_𝒢_MEn.append(𝒢_MEn_sto)
    print(str((1 + ih) * 100 / len(H)) + "%")
print("Model is solved")
print("Time to solve model: " + str(time.time() - start) + " seconds")


########################################
# Part 4: Solve model with the function
start = time.time()
Cstar_det_func, 𝒢_M_det_func, H_Cstar_func, H_𝒢_MEn_func = model.solve(par)
print("Time to solve model: " + str(time.time() - start) + " seconds")


########################################
# Part 5: Comparison of the two solutions
# - The two solutions should be the same
# Cstar_det_func == Cstar_det
print(
    "\nAre the deterministic consumption functions equal? "
    + str(np.allclose(Cstar_det_func, Cstar_det, atol=1e-10))
)
# 𝒢_M_det_func == 𝒢_M_det
print(
    "Are the deterministic cash-on-hand grids equal? "
    + str(np.allclose(𝒢_M_det_func, 𝒢_M_det, atol=1e-10))
)
# H_Cstar_func == H_Cstar
print(
    "Are the stochastic consumption functions equal? "
    + str(np.allclose(H_Cstar_func, H_Cstar, atol=1e-10))
)
