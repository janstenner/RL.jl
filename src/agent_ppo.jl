function create_chain(;ns, na, use_gpu, is_actor, init, nna_scale, drop_middle_layer, fun = relu, tanh_end = true)
    nna_size_actor = Int(floor(10 * nna_scale))
    nna_size_critic = Int(floor(20 * nna_scale))

    if is_actor
        if tanh_end
            if drop_middle_layer
                n = Chain(
                    Dense(ns, nna_size_actor, fun; init = init),
                    Dense(nna_size_actor, na, tanh; init = init)
                )
            else
                n = Chain(
                    Dense(ns, nna_size_actor, fun; init = init),
                    Dense(nna_size_actor, nna_size_actor, fun; init = init),
                    Dense(nna_size_actor, na, tanh; init = init)
                )
            end
        else
            if drop_middle_layer
                n = Chain(
                    Dense(ns, nna_size_actor, fun; init = init),
                    Dense(nna_size_actor, na; init = init)
                )
            else
                n = Chain(
                    Dense(ns, nna_size_actor, fun; init = init),
                    Dense(nna_size_actor, nna_size_actor, fun; init = init),
                    Dense(nna_size_actor, na; init = init)
                )
            end
        end
    else
        if drop_middle_layer
            n = Chain(
                Dense(ns, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        else
            n = Chain(
                Dense(ns, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        end
    end

    model = use_gpu ? n |> gpu : n

    model
end

function create_logσ(;logσ_is_network, ns, na, use_gpu, init, nna_scale, drop_middle_layer, fun = relu, max_σ = 0.3)

    res = nothing

    if logσ_is_network
        res = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = false)
    else
        res = Matrix(Matrix(Float32.(ones(na) .* log(max_σ))')')
    end

    res = use_gpu ? res |> gpu : res

    return res
end

function create_agent_ppo(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, trajectory_length = 1000, learning_rate = 0.00001, fun = relu, fun_critic = nothing, n_envs = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = true, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, clip_grad = 0.5, target_kl = 100.0)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    Agent(
        policy = PPOPolicy(
            approximator = isnothing(approximator) ? ActorCritic(
                actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, max_σ = max_σ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                ),
                critic = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic),
                optimizer_actor = OptimiserChain(ClipNorm(clip_grad), ADAM(learning_rate)),
                optimizer_critic = OptimiserChain(ClipNorm(clip_grad), ADAM(learning_rate)),
            ) : approximator,
            γ = y,
            λ = p,
            clip_range = 0.2f0,
            max_grad_norm = 0.5f0,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            dist = Normal,
            rng = rng,
            update_freq = update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_envs),
                action = Float32 => (size(action_space)[1], n_envs),
                action_log_prob = Float32 => (n_envs),
                reward = Float32 => (n_envs),
                terminal = Bool => (n_envs,),
                next_values = Float32 => (n_envs,),
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
    clip1::Bool
    normalize_advantage::Bool
    start_steps
    start_policy
    target_kl
    last_action_log_prob::Vector{Float32}
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
    rng=Random.GLOBAL_RNG,
    clip1 = false,
    normalize_advantage = normalize_advantage,
    start_steps = -1,
    start_policy = nothing,
    target_kl = 100.0
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
        clip1,
        normalize_advantage,
        start_steps,
        start_policy,
        target_kl,
        [0.0],
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
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::PPOPolicy, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::PPOPolicy)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end


# !!! https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/533/files#r728920324
function (p::PPOPolicy)(env::AbstractEnv)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        dist = prob(p, env)
        action = rand.(p.rng, dist)

        if p.clip1
            clamp!(action, -1.0, 1.0)
        end

        # put the last action log prob behind the clip
        
        ###p.last_action_log_prob = vec(sum(logpdf.(dist, action), dims=1))

        if ndims(action) == 2
            log_p = vec(sum(normlogpdf(dist.μ, dist.σ, action), dims=1))
        else
            log_p = normlogpdf(dist.μ, dist.σ, action)
        end

        p.last_action_log_prob = log_p

        action
    end
end

function (agent::Agent{<:PPOPolicy})(env::MultiThreadEnv)

    if agent.policy.update_step <= policy.start_steps
        agent.policy.start_policy(env)
    else
        dist = prob(agent.policy, env)
        action = rand.(agent.policy.rng, dist)
        if ndims(action) == 2
            action_log_prob = sum(logpdf.(dist, action), dims=1)
        else
            action_log_prob = sum(logpdf.(dist, action), dims=1)
        end
        EnrichedAction(action; action_log_prob=vec(action_log_prob))
    end
end






function update!(
    trajectory::AbstractTrajectory,
    ::PPOPolicy,
    env::MultiThreadEnv,
    ::PreActStage,
    action::EnrichedAction,
)
    push!(
        trajectory;
        state=state(env),
        action=action.action,
        action_log_prob=action.action_log_prob
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(
        trajectory;
        state=state(env),
        action=action,
        action_log_prob=policy.last_action_log_prob
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)

    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
    push!(trajectory[:next_values], policy.approximator.critic(send_to_device(device(policy.approximator), env.state)) |> send_to_host)
end

function update!(
    p::PPOPolicy,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
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
    states = to_device(t[:state])


    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))

    values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)

    # TODO values and next values with n_envs > 1 ???

    advantages = generalized_advantage_estimation(
        t[:reward],
        values,
        t[:next_values],
        γ,
        λ;
        dims=2,
        terminal=t[:terminal]
    )
    returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    advantages = to_device(advantages)

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)

    stop_update = false

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

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

            clamp!(log_p, log(1e-8), 0.0) # clamp old_prob to 1e-5 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            if isnothing(AC.actor_state_tree)
                AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor)
            end

            if isnothing(AC.critic_state_tree)
                AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
            end

            g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic
                v′ = critic(s) |> vec
                if actor isa GaussianNetwork
                    μ, logσ = actor(s)
                    
                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ), a), dims=1))
                    else
                        log_p′ₐ = normlogpdf(μ, exp.(logσ), a)
                    end
                    entropy_loss =
                        mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                else
                    # actor is assumed to return discrete logits
                    logit′ = actor(s)

                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)
                    log_p′ₐ = log_p′[a]
                    entropy_loss = -sum(p′ .* log_p′) * 1 // size(p′, 2)
                end
                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl
                        println("Target KL overstepped: $(approx_kl_div)")
                        stop_update = true
                    end
                end

                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))
                critic_loss = mean((r .- v′) .^ 2)
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                loss
            end
            
            if !stop_update
                Flux.update!(AC.actor_state_tree, AC.actor, g_actor)
                Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
            else
                break
            end

        end

        if stop_update
            break
        end
    end
end

