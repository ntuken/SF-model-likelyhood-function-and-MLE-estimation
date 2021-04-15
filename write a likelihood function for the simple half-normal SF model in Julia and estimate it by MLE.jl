using CSV, DataFrames
using Optim, NLSolversBase, Random
using Distributions
using LinearAlgebra, GLM
using Plots

#  Read in Dataset
df = CSV.read("sampledata.csv", DataFrame; header=1, delim=",")

#  append a column of 1 to be used as a constant
df[!, :_cons] .=1;



n = size(df, 1)     # n is number of observations
nvar = size(df, 2)  # nvar is number of variable 

describe(df)

# get sfmodel maximum likelyhood function
function sfmodel_spec(dist, depvar, frontier, log_σ_u_ED, β, log_σ_v, log_σ_u_ME)
    """
    dist: distribution of u_i and v_i
    depvar: depend variable with matrix or Dataframe type
    frontier: independent variable with matrix or Dataframe type
    log_σ_u_ED: exogenous determinant of log_σ_u with matrix or Dataframe type
    β: marginal effect for depvar vector with length equal to columns of frontier with matrix or DataFrame type
    log_σ_v: float type
    log_σ_u_ME: marginal effect for log_σ_u_ED with length equal to columns of log_σ_u_ED
    """ 
    
    
    if dist == "half"
        d = Normal(0, 1)
    end
    
    X = convert(Matrix, frontier)
    Y = vec(depvar)
    W_u = convert(Matrix, log_σ_u_ED)
    ε = vec(Y - X*β)  #  ε is residual vector  !! some error occur
    
    n = size(X, 1)  # n is number of observation
    
    σ_u = map(exp, (W_u*log_σ_u_ME))  # σ_u is vector contain each observation's variance of efficiency
    σ_v = exp(log_σ_v)
    
    tmp = vec(map(σ_ui -> 1/2*log(σ_ui^2 + σ_v), σ_u))
    u_star = vec((-σ_u.^2 .*ε)./(σ_v^2 .+ σ_u.^2))
    sigma_star = vec(map(σ_ui -> (σ_v^2*σ_ui^2)/(σ_v^2 + σ_ui^2), σ_u))
    
    tmp_2 = vec(map(t -> log(pdf(d, t)), ε./(((σ_u.^2 .+ σ_v^2)).^(1/2))))
    
    
    llike_i = -log(1/2) .- tmp .+ tmp_2 .+ map(t -> log(cdf(d, t)), u_star./(sigma_star) .^(1/2))
    llike = sum(-1 .*llike_i)  # sum up llike_i
    
end


# take imput data

y = df.yvar
x = df[:,["Lland", "PIland", "Llabor", "Lbull", "Lcost", "yr", "_cons"]]
log_σ_u_ED = df[:, ["age", "school", "yr", "_cons"]]


nvar = size(x, 2)    # number of xvar

n_ED = size(log_σ_u_ED, 2)  # number of exogenous determinant of log_σ_u

# since β is consistent in both OLS and SF model, I run OLS to get β coefficient as initial value

fm = @formula(yvar ~ Lland + PIland + Llabor + Lbull + Lcost + yr + _cons)
linearRegressor = lm(fm, df[:,["yvar", "Lland", "PIland", "Llabor", "Lbull", "Lcost", "yr", "_cons"]])
print(linearRegressor)


initial_value = [0.273592, 0.108956, 1.26251, -0.50067, 0.00177811, 0.049151, 0.877344, 0.108956, 0.108956, 0.108956, 0.108956, 0.108956]

func = TwiceDifferentiable(vars -> sfmodel_spec("half", y, x, log_σ_u_ED,
                                                              vars[1:nvar], vars[nvar+1], vars[nvar+2:nvar+n_ED+1]),
                                                              initial_value; autodiff=:forward);
opt = optimize(func, initial_value)


parameters = Optim.minimizer(opt)
parameters[nvar+1] = exp(parameters[nvar+1])  # σ_v estimation

# build up β_table
β = parameters[1:nvar]
frontier_name = ["Lland", "PIland", "Llabor", "Lbull", "Lcost", "yr", "_cons"]
β_table = convert(DataFrame, [frontier_name β])
rename!(β_table, ["variable_name", "β"])
println(β_table)


ED_names = ["age", "school", "yr", "_cons"]
log_σ_u_ED_table = convert(DataFrame, [ED_names parameters[nvar+2:end]])
rename!(log_σ_u_ED_table, ["variable_name", "coefficient"])
println(log_σ_u_ED_table)


# build up efficiency expectation table

dist = Normal(0, 1)
σ_u = map(t->exp(t), convert(Matrix, df[:, ED_names]) * parameters[nvar+2:end])  # vector of estimation of σ_u
σ_v = parameters[nvar+1]
β = parameters[1:nvar]
x = convert(Matrix, x) 
ε = vec(y - x*β)

u_star = vec((-σ_u.^2 .*ε)./(σ_v^2 .+ σ_u.^2))
sigma_star = vec(map(σ_ui -> (σ_v^2*σ_ui^2)/(σ_v^2 + σ_ui^2), σ_u))
tmp = u_star ./(sigma_star .^(1/2))

# Exp_u_i is expectation of u_i condition on ε_i
Exp_u_i = sigma_star .^(1/2) .*map(t->pdf(dist, t), tmp) ./(map(t->cdf(dist, t), tmp)) .+ u_star   

histogram(vec(Exp_u_i), bins=100)


# build up efficiency variance density plots

histogram(vec(σ_u), bins=100)


