

Base.@kwdef struct CustomNeuralNetworkApproximator{M,O}
    model::M
    optimizer::O = nothing
end

# some model may accept multiple inputs
(app::CustomNeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

@forward CustomNeuralNetworkApproximator.model Flux.testmode!,
Flux.trainmode!,
Flux.params,
device


update!(app::CustomNeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)

copyto!(dest::CustomNeuralNetworkApproximator, src::CustomNeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, Flux.params(src))







#####
# ActorCritic
#####

"""
    ActorCritic(;actor, critic, optimizer=ADAM())

The `actor` part must return logits (*Do not use softmax in the last layer!*), and the `critic` part must return a state value.
"""
Base.@kwdef struct ActorCritic{A,C,O}
    actor::A
    critic::C
    optimizer::O = ADAM()
end

functor(x::ActorCritic) =
    (actor=x.actor, critic=x.critic), y -> ActorCritic(y.actor, y.critic, x.optimizer)

update!(app::ActorCritic, gs) = Flux.Optimise.update!(app.optimizer, params(app), gs)

function Base.copyto!(dest::ActorCritic, src::ActorCritic)
    Flux.loadparams!(dest.actor, params(src.actor))
    Flux.loadparams!(dest.critic, params(src.critic))
end

@forward ActorCritic.critic device




#####
# GaussianNetwork (for PPO)
##### 

"""
     normlogpdf(μ, σ, x; ϵ = 1.0f-8)
GPU automatic differentiable version for the logpdf function of normal distributions.
Adding an epsilon value to guarantee numeric stability if sigma is exactly zero
(e.g. if relu is used in output layer).
"""
const log2π = log(2f0π)

function normlogpdf(μ, σ, x; ϵ = 1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end


"""
    GaussianNetwork(;pre=identity, μ, logσ, min_σ=0f0, max_σ=Inf32, normalizer = tanh)

Returns `μ` and `logσ` when called.  Create a distribution to sample from using
`Normal.(μ, exp.(logσ))`. `min_σ` and `max_σ` are used to clip the output from
`logσ`. Actions are normalized according to the specified normalizer function.
"""
Base.@kwdef struct GaussianNetwork{P,U,S,F}
    pre::P = identity
    μ::U
    logσ::S
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
    normalizer::F = tanh
end

GaussianNetwork(pre, μ, logσ, normalizer=tanh) = GaussianNetwork(pre, μ, logσ, 0.0f0, Inf32, normalizer)

Flux.@functor GaussianNetwork


"""
This function is compatible with a multidimensional action space. When outputting an action, it uses the `normalizer` function to normalize it elementwise.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    x = model.pre(s)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    if is_sampling
        σ = exp.(logσ)
        z = Zygote.ignore() do
            noise = randn(rng, Float32, size(μ))
            model.normalizer.(μ .+ σ .* noise)
        end
        if is_return_log_prob
            logp_π = sum(normlogpdf(μ, σ, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims=1)
            return z, logp_π
        else
            return z
        end
    else
        return μ, logσ
    end
end

"""
    (model::GaussianNetwork)(rng::AbstractRNG, state, action_samples::Int)
Sample `action_samples` actions from each state. Returns a 3D tensor with dimensions (action_size x action_samples x batch_size).
`state` must be 3D tensor with dimensions (state_size x 1 x batch_size). Always returns the logpdf of each action along.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s, action_samples::Int)
    x = model.pre(s)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))

    σ = exp.(logσ)
    z = Zygote.ignore() do
        noise = randn(rng, Float32, (size(μ, 1), action_samples, size(μ, 3))...)
        model.normalizer.(μ .+ σ .* noise)
    end
    logp_π = sum(normlogpdf(μ, σ, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims=1)
    return z, logp_π
end

function (model::GaussianNetwork)(state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    model(Random.GLOBAL_RNG, state; is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
end

function (model::GaussianNetwork)(state, action_samples::Int)
    model(Random.GLOBAL_RNG, state, action_samples)
end

function (model::GaussianNetwork)(state, action)
    x = model.pre(state)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    σ = exp.(logσ)
    logp_π = sum(normlogpdf(μ, σ, action) .- (2.0f0 .* (log(2.0f0) .- action .- softplus.(-2.0f0 .* action))), dims=1)
    return logp_π
end