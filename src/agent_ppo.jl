function create_chain(;na, ns, use_gpu, is_actor, init, copyfrom = nothing, nna_scale, drop_middle_layer, fun = relu)
    nna_size_actor = Int(floor(10 * nna_scale))
    nna_size_critic = Int(floor(20 * nna_scale))

    if is_actor
        if drop_middle_layer
            n = Chain(
                Dense(ns, nna_size_actor, fun; init = init),
                Dense(nna_size_actor, na, tanh; init = init),
            )
        else
            n = Chain(
                Dense(ns, nna_size_actor, fun; init = init),
                Dense(nna_size_actor, nna_size_actor, fun; init = init),
                Dense(nna_size_actor, na, tanh; init = init),
            )
        end
    else
        if drop_middle_layer
            n = Chain(
                Dense(ns + na, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        else
            n = Chain(
                Dense(ns + na, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        end
    end

    n
end

function create_agent_ppo(;action_space, state_space, use_gpu, rng, y, p, update_freq = 128, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, trajectory_length = 1000, mono = false, learning_rate = 0.00001, fun = relu, fun_critic = nothing, n_envs = 2)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)

    init = Flux.glorot_uniform(rng)

    if mono
        reward_size = 1
    else
        reward_size = size(action_space)[1]
    end

    Agent(
        policy = PPOPolicy(
            approximator = ActorCritic(
                actor = GaussianNetwork(
                    pre = Chain(
                        Dense(size(state_space)[1], 64, relu; init = init),
                        Dense(64, 64, relu; init = init),
                    ),
                    μ = Chain(Dense(64, 4, tanh; init = init), vec),
                    logσ = Chain(Dense(64, 4; init = init), vec),
                ),
                critic = Chain(
                    Dense(size(state_space)[1], 64, relu; init = init),
                    Dense(64, 64, relu; init = init),
                    Dense(64, 1; init = init),
                ),
                optimizer = Flux.ADAM(learning_rate),
            ),
            γ = y,
            λ = p,
            clip_range = 0.2f0,
            max_grad_norm = 0.5f0,
            n_epochs = 4,
            n_microbatches = 32,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.00f0,
            dist = Normal,
            rng = rng,
            update_freq = update_freq,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_envs),
                action = Float32 => (size(action_space)[1], n_envs),
                action_log_prob = Float32 => (size(action_space)[1], n_envs),
                reward = Float32 => (reward_size, n_envs,),
                terminal = Bool => (n_envs,),
                value = Float32 => (n_envs,),
        ),
    )
end



"""
    PPOPolicy(;kwargs)

# Keyword arguments

- `approximator`,
- `γ = 0.99f0`,
- `λ = 0.95f0`,
- `clip_range = 0.2f0`,
- `max_grad_norm = 0.5f0`,
- `n_microbatches = 4`,
- `n_epochs = 4`,
- `actor_loss_weight = 1.0f0`,
- `critic_loss_weight = 0.5f0`,
- `entropy_loss_weight = 0.01f0`,
- `dist = Categorical`,
- `rng = Random.GLOBAL_RNG`,

If `dist` is set to `Categorical`, it means it will only work
on environments of discrete actions. To work with environments of continuous
actions `dist` should be set to `Normal` and the `actor` in the `approximator`
should be a `GaussianNetwork`. Using it with a `GaussianNetwork` supports 
multi-dimensional action spaces, though it only supports it under the assumption
that the dimensions are independent since the `GaussianNetwork` outputs a single
`μ` and `σ` for each dimension which is used to simplify the calculations.
"""

const Normal = 1
const Categorical = 2

mutable struct PPOPolicy{A<:ActorCritic,D,R} <: AbstractPolicy
    approximator::A
    γ::Float32
    λ::Float32
    clip_range::Float32
    max_grad_norm::Float32
    n_microbatches::Int
    n_epochs::Int
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    rng::R
    n_random_start::Int
    update_freq::Int
    update_step::Int
    # for logging
    norm::Matrix{Float32}
    actor_loss::Matrix{Float32}
    critic_loss::Matrix{Float32}
    entropy_loss::Matrix{Float32}
    loss::Matrix{Float32}
end

function PPOPolicy(;
    approximator,
    update_freq,
    n_random_start=0,
    update_step=0,
    γ=0.99f0,
    λ=0.95f0,
    clip_range=0.2f0,
    max_grad_norm=0.5f0,
    n_microbatches=4,
    n_epochs=4,
    actor_loss_weight=1.0f0,
    critic_loss_weight=0.5f0,
    entropy_loss_weight=0.01f0,
    dist=Normal,
    rng=Random.GLOBAL_RNG
)
    PPOPolicy{typeof(approximator),dist,typeof(rng)}(
        approximator,
        γ,
        λ,
        clip_range,
        max_grad_norm,
        n_microbatches,
        n_epochs,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        rng,
        n_random_start,
        update_freq,
        update_step,
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
    )
end

function prob(
    p::PPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal},
    state::AbstractArray,
    mask,
)
    if p.update_step < p.n_random_start
        @error "todo"
    else
        μ, logσ =
            p.approximator.actor(send_to_device(device(p.approximator), state)) |>
            send_to_host
        StructArray{Normal}((μ, exp.(logσ)))
    end
end

function prob(p::PPOPolicy{<:ActorCritic,Categorical}, state::AbstractArray, mask)
    logits = p.approximator.actor(send_to_device(device(p.approximator), state))
    if !isnothing(mask)
        logits .+= ifelse.(mask, 0.0f0, typemin(Float32))
    end
    logits = logits |> softmax |> send_to_host
    if p.update_step < p.n_random_start
        [
            Categorical(fill(1 / length(x), length(x)); check_args=false) for
            x in eachcol(logits)
        ]
    else
        [Categorical(x; check_args=false) for x in eachcol(logits)]
    end
end

function prob(p::PPOPolicy, env::MultiThreadEnv)
    mask = ActionStyle(env) === FULL_ACTION_SET ? legal_action_space_mask(env) : nothing
    prob(p, state(env), mask)
end

function prob(p::PPOPolicy, env::AbstractEnv)
    s = state(env)
    s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = ActionStyle(env) === FULL_ACTION_SET ? legal_action_space_mask(env) : nothing
    prob(p, s, mask)
end

(p::PPOPolicy)(env::MultiThreadEnv) = rand.(p.rng, prob(p, env))

# !!! https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/533/files#r728920324
(p::PPOPolicy)(env::AbstractEnv) = rand.(p.rng, prob(p, env))

function (agent::Agent{<:PPOPolicy})(env::MultiThreadEnv)
    dist = prob(agent.policy, env)
    action = rand.(agent.policy.rng, dist)
    if ndims(action) == 2
        action_log_prob = sum(logpdf.(dist, action), dims=1)
    else
        action_log_prob = logpdf.(dist, action)
    end
    EnrichedAction(action; action_log_prob=vec(action_log_prob))
end

function update!(
    p::PPOPolicy,
    t::Union{PPOTrajectory,MaskedPPOTrajectory},
    ::AbstractEnv,
    ::PreActStage,
)
    length(t) == 0 && return  # in the first update, only state & action are inserted into trajectory
    p.update_step += 1
    if p.update_step % p.update_freq == 0
        _update!(p, t)
    end
end

function _update!(p::PPOPolicy, t::Any)
    rng = p.rng
    AC = p.approximator
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight
    D = device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_rollout = size(t[:terminal])
    @assert n_envs * n_rollout % n_microbatches == 0 "size mismatch"
    microbatch_size = n_envs * n_rollout ÷ n_microbatches

    n = length(t)
    states_plus = to_device(t[:state])

    if t isa MaskedPPOTrajectory
        LAM = to_device(t[:legal_actions_mask])
    end

    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))
    states_plus_values =
        reshape(send_to_host(AC.critic(flatten_batch(states_plus))), n_envs, :)

    # TODO: make generalized_advantage_estimation GPU friendly
    advantages = generalized_advantage_estimation(
        t[:reward],
        states_plus_values,
        γ,
        λ;
        dims=2,
        terminal=t[:terminal]
    )
    returns = to_device(advantages .+ select_last_dim(states_plus_values, 1:n_rollout))
    advantages = to_device(advantages)

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)

    # TODO: normalize advantage
    for epoch in 1:n_epochs
        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches
            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]
            if t isa MaskedPPOTrajectory
                lam = select_last_dim(flatten_batch(select_last_dim(LAM, 2:n+1)), inds)

            else
                lam = nothing
            end

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds]
            adv = vec(advantages)[inds]

            ps = Flux.params(AC)
            gs = gradient(ps) do
                v′ = AC.critic(s) |> vec
                if AC.actor isa GaussianNetwork
                    μ, logσ = AC.actor(s)
                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ), a), dims=1))
                    else
                        log_p′ₐ = normlogpdf(μ, exp.(logσ), a)
                    end
                    entropy_loss =
                        mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                else
                    # actor is assumed to return discrete logits
                    raw_logit′ = AC.actor(s)
                    if isnothing(lam)
                        logit′ = raw_logit′
                    else
                        logit′ = raw_logit′ .+ ifelse.(lam, 0.0f0, typemin(Float32))
                    end
                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)
                    log_p′ₐ = log_p′[a]
                    entropy_loss = -sum(p′ .* log_p′) * 1 // size(p′, 2)
                end
                ratio = exp.(log_p′ₐ .- log_p)
                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))
                critic_loss = mean((r .- v′) .^ 2)
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                ignore_derivatives() do
                    p.actor_loss[i, epoch] = actor_loss
                    p.critic_loss[i, epoch] = critic_loss
                    p.entropy_loss[i, epoch] = entropy_loss
                    p.loss[i, epoch] = loss
                end

                loss
            end

            p.norm[i, epoch] = clip_by_global_norm!(gs, ps, p.max_grad_norm)
            update!(AC, gs)
        end
    end
end

function update!(
    trajectory::Union{PPOTrajectory,MaskedPPOTrajectory},
    ::PPOPolicy,
    env::MultiThreadEnv,
    ::PreActStage,
    action::EnrichedAction,
)
    push!(
        trajectory;
        state=state(env),
        action=action.action,
        action_log_prob=action.meta.action_log_prob
    )

    if trajectory isa MaskedPPOTrajectory
        push!(trajectory; legal_actions_mask=legal_action_space_mask(env))
    end
end
