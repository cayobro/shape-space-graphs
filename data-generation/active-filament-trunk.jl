using ActiveFilaments
using StaticArrays
using Plots
using GLMakie
using CSV
using Distances
using JLD2
using NPZ
using JSON3


#region Parameters and filament geometry
# Filament geometry
L = 0.09 # m
R2 = L / 20.0
R1 = R2 * 5.0 / 6.0

# Fiber architectures
M = 3
ΩDeg = [-108, 108, 0] # In degrees
Ω = ΩDeg * 2 * pi / 360 # Fiber revolution

# Fiber activation sectors
σ = 48.0 * 2 * pi / 360
θ0 = [90.0 - 48.0 / 2, 90.0 + 48.0 / 2, 270] * 2 * pi / 360

# Fiber architectures in the three rings
α2 = [atan.(R2 / L * Ωi) for Ωi in Ω]

# Mechanical properties
E_acr = 0.1278e6
ν_acr = 0.5
E_LCE = 1.4e6
E = E_LCE * σ / (2 * pi) # Stiffness correction for overlapping rings (see paper)
ν_LCE = 0.495

# Build the rings
rings = [
        Ring(MechanicalProperties(E, ν_LCE), Geometry(R1, R2), FiberArchitecture(α2[1])),
        Ring(MechanicalProperties(E, ν_LCE), Geometry(R1, R2), FiberArchitecture(α2[2])),
        Ring(MechanicalProperties(E, ν_LCE), Geometry(R1, R2), FiberArchitecture(α2[3]))
        ]

# Build the filament
filament = AFilament(
    rings; 
    L = L, 
    ρvol = 1000.0, 
    innerTube = InnerTube(MechanicalProperties(E_acr, ν_acr), Geometry(0.0, R1)), 
    expression = Val{false}
    )
#endregion

#region Set up the BVP
r0 = [0.0, 0.0, 0.0]    # Boundary condition on r(Z = 0)
d10 = [1.0, 0.0, 0.0]   # Boundary condition on d1(Z = 0)
d20 = [0.0, 1.0, 0.0]   # Boundary condition on d2(Z = 0)
d30 = [0.0, 0.0, 1.0]   # Boundary condition on d3(Z = 0)
bcs = SVector{12, Float64}([r0..., d10..., d20..., d30...]) # Boundary conditions
m0 = [0.0, 0.0, 0.0]    # Initial guess for the momoent at Z = 0
u0 = [bcs..., m0...]    # A constant initial guess for the BVP solver
g = -9.8                # Gravitational acceleration
Ng = 1                  # Number of iterations for g
g_range = range(start = 0.0, stop = g, length = Ng + 1)[2:end] # Exclude g = 0
#endregion


nTrajectories = 1000
path = '/'

γ = [-1.2, -0.7, -0.5] # Actual activations
γ *= M                 # Unify the gamma scaling for overlapping rings
activation_type = [ActivationPiecewiseGamma(1, [γ[i]], σ, θ0[i]) for i in 1:M]

γBounds = [
    ([-5.0 / 3.0] * M, [0.0]), 
    ([-5.0 / 3.0] * M, [0.0]),
    ([-5.0 / 3.0] * M, [0])] # M is the number of fibers

(activationsFourier, activationsGamma) = generateRandomActivations(
    activation_type, γBounds, M, nTrajectories
    )
    
u_data = []
z_data = []
gamma_data = []
for i in 1:nTrajectories
    act = activationsGamma[i]
    sol = selfWeightSolve(filament, act, u0, bcs, g_range, solver = 1)
    # extract gamma num
    γ_scaled = [act[i].γ[1] for i in 1:M]  # extract the inner γ values
    γ= γ_scaled ./ M  # divide by M to undo the scaling
    push!(gamma_data, γ)
    # extract u vec
    u = sol.u
    push!(u_data, u)
    # extract z vec
    z = sol.t
    push!(z_data, z)
end

# Step 1: Convert u_data into a list of nz_i×15 matrices (nested arrays)
u_clean = [
    [collect(Float64.(v)) for v in u_row] for u_row in u_data
]  # u_clean[i] = list of 15D vectors

# Step 2: Convert z_data and gamma_data into regular float vectors
z_clean = [collect(Float64.(z)) for z in z_data]
gamma_clean = [collect(Float64.(g)) for g in gamma_data]

# Step 3: Package everything in a Dict
data_dict = Dict(
    "u" => u_clean,
    "z" => z_clean,
    "gamma" => gamma_clean
)

# Step 4: Write to JSON
open(string(path,"_data.json"), "w") do io
    JSON3.write(io, data_dict)
end



