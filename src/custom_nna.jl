

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
    logσ_is_network::Bool = false
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
    normalizer::F = tanh
end

GaussianNetwork(pre, μ, logσ, min_σ=0.0f0, max_σ=Inf32, normalizer=tanh) = GaussianNetwork(pre, μ, logσ, min_σ, max_σ, normalizer)

Flux.@functor GaussianNetwork


"""
This function is compatible with a multidimensional action space. When outputting an action, it uses the `normalizer` function to normalize it elementwise.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s; is_sampling::Bool=false, is_return_log_prob::Bool=false)

    x = model.pre(s)

    if model.logσ_is_network
        μ, raw_logσ = model.μ(x), model.logσ(x)
    else
        μ, raw_logσ = model.μ(x), model.logσ
        # the first method leads to Cuda complaining about scalar indexing
        # raw_logσ = repeat(raw_logσ, outer=(1,size(μ)[2]))
        if ndims(μ) >= 2
            raw_logσ = hcat([raw_logσ for i in 1:size(μ)[2]]...)
        else
            raw_logσ = raw_logσ[:]
        end
    end

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
        return μ, deepcopy(logσ)
    end
end

function (model::GaussianNetwork)(state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    model(Random.GLOBAL_RNG, state; is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
end