# Recurrent extension of agent_sac.jl. Load into RL with Base.include(RL, path).
# This file deliberately contains its own networks, sequence sampler and stage
# methods so that the existing SAC, trajectory and run implementations stay intact.
export SACLSTMPolicy, create_agent_sac_lstm, reset_sac_lstm!, observe_sac_lstm!,
       act_sac_lstm!, sample_sac_lstm, sac_lstm_history, sac_lstm_actions,
       sac_lstm_qvalues, sac_lstm_episode_starts

struct SACLSTMEncoder{O,A,R,L}
    observ_embedder::O
    action_embedder::A
    reward_embedder::R
    lstm::L
end
Flux.@layer SACLSTMEncoder

struct SACLSTMActor{E,S,H}
    encoder::E
    shortcut::S
    head::H
end
Flux.@layer SACLSTMActor

struct SACLSTMCritic{E,S,Q1,Q2}
    encoder::E
    shortcut::S
    qnetwork1::Q1
    qnetwork2::Q2
end
Flux.@layer SACLSTMCritic

_sac_lstm_embed(layer, x) = layer(x)
_sac_lstm_embed(::Nothing, x) = selectdim(x, 1, 1:0)

function _sac_lstm_input(encoder::SACLSTMEncoder, obs, prev_actions, prev_rewards)
    vcat(_sac_lstm_embed(encoder.action_embedder, prev_actions),
         _sac_lstm_embed(encoder.reward_embedder, prev_rewards),
         encoder.observ_embedder(obs))
end

"""Reconstruct hidden states from zero, with arrays in (features, batch, time) order."""
function sac_lstm_history(encoder::SACLSTMEncoder, obs, prev_actions, prev_rewards)
    x = _sac_lstm_input(encoder, obs, prev_actions, prev_rewards)
    # Flux.LSTM expects (features, time, batch); no mutable rollout state is used.
    permutedims(encoder.lstm(permutedims(x, (1, 3, 2))), (1, 3, 2))
end

function _sac_lstm_step(encoder::SACLSTMEncoder, obs, prev_actions, prev_rewards, hc)
    x = _sac_lstm_input(encoder, obs, prev_actions, prev_rewards)
    cell = encoder.lstm.cell
    cell(x, isnothing(hc) ? Flux.initialstates(cell) : hc)
end

# Keep the GaussianNetwork conventions of SAC, including its trainable constant
# log-sigma option and stable tanh Jacobian. Transfer RNG noise explicitly so the
# same implementation works with a StableRNG and CPU or CUDA network parameters.
function _sac_lstm_draw(head::GaussianNetwork, rng, features; deterministic=false)
    mu, log_sigma = head(features)
    if deterministic
        return head.normalizer.(mu), nothing
    end
    noise = ignore_derivatives() do
        send_to_device(device(head), randn(rng, Float32, size(mu)))
    end
    sigma = exp.(log_sigma)
    u = mu .+ sigma .* noise
    actions = head.normalizer.(u)
    log_pi = sum(normlogpdf(mu, sigma, u) .- log_jac_tanh(u); dims=1)
    return actions, log_pi
end

function sac_lstm_actions(actor::SACLSTMActor, obs, prev_actions, prev_rewards, rng;
                          deterministic=false)
    hidden = sac_lstm_history(actor.encoder, obs, prev_actions, prev_rewards)
    features = vcat(hidden, actor.shortcut(obs))
    b, len = size(obs, 2), size(obs, 3)
    actions, log_pi = _sac_lstm_draw(actor.head, rng,
        reshape(features, size(features, 1), :); deterministic)
    return reshape(actions, size(actions, 1), b, len),
           isnothing(log_pi) ? nothing : reshape(log_pi, 1, b, len)
end

function _sac_lstm_qheads(critic::SACLSTMCritic, hidden, obs, current_actions)
    features = vcat(hidden, critic.shortcut(vcat(obs, current_actions)))
    critic.qnetwork1(features), critic.qnetwork2(features)
end

function sac_lstm_qvalues(critic::SACLSTMCritic, obs, prev_actions, prev_rewards,
                         current_actions)
    hidden = sac_lstm_history(critic.encoder, obs, prev_actions, prev_rewards)
    len = size(current_actions, 3)
    _sac_lstm_qheads(critic, hidden[:, :, 1:len], obs[:, :, 1:len], current_actions)
end

Base.@kwdef mutable struct SACLSTMPolicy <: AbstractPolicy
    actor::SACLSTMActor
    critic::SACLSTMCritic
    target_critic::SACLSTMCritic
    optimizer_actor
    optimizer_critic
    actor_state_tree = nothing
    critic_state_tree = nothing
    action_space::Space
    state_space::Space
    γ::Float32 = 0.99f0
    τ::Float32 = 0.005f0
    α::Float32 = 0.2f0
    log_α = [log(0.2f0)]
    optimizer_log_α
    log_α_state_tree = nothing
    batch_size::Int = 32
    sequence_length::Int = 64
    episode_starts_only::Bool = false
    start_steps::Int = -1
    start_policy = nothing
    update_after::Int = 1000
    update_freq::Int = 50
    update_loops::Int = 1
    automatic_entropy_tuning::Bool = true
    lr_alpha::Float32 = 0.0003f0
    target_entropy::Float32 = -1.0f0
    update_step::Int = 0
    gradient_step::Int = 0
    rng = Random.GLOBAL_RNG
    use_popart::Bool = false
    verbose::Bool = false
    # weight of the BC term in IL updates (TD3+BC style); nothing = off
    bc_alpha::Union{Nothing,Float32} = nothing
    last_bc_loss::Float32 = 0.0f0
    # share of expert samples accepted by the Q-filter in the last BC update
    last_bc_accept::Float32 = 0.0f0
    # Rollout memory is separate from trainable parameters and replay data.
    actor_internal_state = nothing
    prev_action = nothing
    prev_reward = nothing
    last_action = nothing
    episode_start_pending::Bool = true
    last_reward_term::Float32 = 0.0f0
    last_entropy_term::Float32 = 0.0f0
    last_actor_loss::Float32 = 0.0f0
    last_critic1_loss::Float32 = 0.0f0
    last_critic2_loss::Float32 = 0.0f0
    last_q1_mean::Float32 = 0.0f0
    last_q2_mean::Float32 = 0.0f0
    last_target_q_mean::Float32 = 0.0f0
    last_mean_minus_log_pi::Float32 = 0.0f0
end

function _create_sac_lstm_encoder(ns, na, obs_size, act_size, rew_size, hidden_size, init)
    SACLSTMEncoder(
        Dense(ns => obs_size, relu; init),
        act_size == 0 ? nothing : Dense(na => act_size, relu; init),
        rew_size == 0 ? nothing : Dense(1 => rew_size, relu; init),
        Flux.LSTM((obs_size + act_size + rew_size) => hidden_size;
                  init_kernel=init, init_recurrent_kernel=init))
end

"""
    create_agent_sac_lstm(; action_space, state_space, rng, y, ...)

SAC with a separate recurrent actor and recurrent twin-Q critic. The Q heads
share their history encoder, as in pomdp-baselines. Existing SAC head sizes,
optimizers, entropy tuning and update scheduling are preserved. New defaults:
32/16/16 observation/action/reward embeddings, 64 transitions per sequence,
and LSTM widths equal to the corresponding original MLP hidden widths.

`batch_size` counts sequences. Each sampled sequence starts with zero memory
and zero previous action/reward. With `episode_starts_only=true`, sequences
start only at stored episode beginnings (as pomdp-baselines with
`sampled_seq_len = -1` when `sequence_length` covers a whole episode), so the
zero initial memory matches inference. Terminal and truncated transitions both end
sequences; only terminated transitions suppress bootstrapping. `update_IL`
runs the same SAC updates on ordered external SARTTS trajectories.
"""
function create_agent_sac_lstm(; action_space, state_space, use_gpu=false, rng, y,
    t=0.005f0, a=0.2f0, nna_scale=1, nna_scale_critic=nothing,
    network_depth=2, network_depth_critic=nothing, drop_middle_layer=nothing,
    drop_middle_layer_critic=nothing, learning_rate=0.00001,
    learning_rate_critic=nothing, fun=gelu, fun_critic=nothing, tanh_end=false,
    n_agents=1, logσ_is_network=false, batch_size=32, start_steps=-1,
    start_policy=nothing, update_after=1000, update_freq=50, update_loops=1,
    max_σ=2.0f0, clip_grad=0.5, start_logσ=0.0, betas=(0.9, 0.999),
    trajectory_length=10_000, automatic_entropy_tuning=true, lr_alpha=nothing,
    target_entropy=nothing, use_popart=false, verbose=false,
    observ_embedding_size=32, action_embedding_size=16, reward_embedding_size=16,
    rnn_hidden_size=nothing, rnn_hidden_size_critic=nothing, sequence_length=64,
    episode_starts_only=false, bc_alpha=nothing)

    isnothing(nna_scale_critic) && (nna_scale_critic = nna_scale)
    !isnothing(drop_middle_layer) && (network_depth = drop_middle_layer ? 1 : 2)
    !isnothing(drop_middle_layer_critic) &&
        (network_depth_critic = drop_middle_layer_critic ? 1 : 2)
    isnothing(network_depth_critic) && (network_depth_critic = network_depth)
    network_depth, network_depth_critic = max(1, Int(network_depth)), max(1, Int(network_depth_critic))
    isnothing(fun_critic) && (fun_critic = fun)
    isnothing(learning_rate_critic) && (learning_rate_critic = learning_rate)
    ns, na = size(state_space)[1], size(action_space)[1]
    isnothing(target_entropy) && (target_entropy = -Float32(na))
    isnothing(lr_alpha) && (lr_alpha = Float32(learning_rate))
    isnothing(rnn_hidden_size) && (rnn_hidden_size = floor(Int, 10 * nna_scale))
    isnothing(rnn_hidden_size_critic) && (rnn_hidden_size_critic = floor(Int, 20 * nna_scale_critic))
    observ_embedding_size > 0 || throw(ArgumentError("observ_embedding_size must be positive"))
    min(action_embedding_size, reward_embedding_size) >= 0 || throw(ArgumentError("embedding sizes must be nonnegative"))
    min(rnn_hidden_size, rnn_hidden_size_critic, batch_size, sequence_length, n_agents,
        trajectory_length, update_freq, update_loops) > 0 || throw(ArgumentError("sizes and update intervals must be positive"))
    a > 0 || throw(ArgumentError("entropy temperature must be positive"))
    init = Flux.glorot_uniform(rng)
    actor_input = rnn_hidden_size + observ_embedding_size
    actor = SACLSTMActor(
        _create_sac_lstm_encoder(ns, na, observ_embedding_size, action_embedding_size,
                                reward_embedding_size, rnn_hidden_size, init),
        Dense(ns => observ_embedding_size, relu; init),
        GaussianNetwork(
            μ=create_chain(ns=actor_input, na=na, use_gpu=false, is_actor=true,
                init=init, nna_scale=nna_scale, network_depth=network_depth,
                fun=fun, tanh_end=tanh_end),
            logσ=create_logσ(logσ_is_network=logσ_is_network, ns=actor_input, na=na,
                use_gpu=false, init=init, nna_scale=nna_scale, network_depth=network_depth,
                fun=fun, start_logσ=start_logσ),
            logσ_is_network=logσ_is_network, max_σ=max_σ))
    shortcut_size = observ_embedding_size + action_embedding_size + reward_embedding_size
    make_q() = create_critic_PPO2(ns=rnn_hidden_size_critic + shortcut_size, na=0,
        use_gpu=false, init=init, nna_scale=nna_scale_critic,
        network_depth=network_depth_critic, fun=fun_critic, popart=use_popart)
    critic = SACLSTMCritic(
        _create_sac_lstm_encoder(ns, na, observ_embedding_size, action_embedding_size,
                                reward_embedding_size, rnn_hidden_size_critic, init),
        Dense((ns + na) => shortcut_size, relu; init), make_q(), make_q())
    if use_gpu
        actor, critic = Flux.gpu(actor), Flux.gpu(critic)
    end
    policy = SACLSTMPolicy(; actor, critic, target_critic=deepcopy(critic),
        optimizer_actor=Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
        optimizer_critic=Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas)),
        action_space, state_space, γ=Float32(y), τ=Float32(t), α=Float32(a),
        log_α=Float32[log(a)], optimizer_log_α=Optimisers.Adam(lr_alpha),
        batch_size, sequence_length, episode_starts_only, start_steps, start_policy, update_after,
        update_freq, update_loops, automatic_entropy_tuning, lr_alpha=Float32(lr_alpha),
        target_entropy=Float32(target_entropy), rng, use_popart, verbose,
        bc_alpha=isnothing(bc_alpha) ? nothing : Float32(bc_alpha))
    reset_sac_lstm!(policy; batch_size=n_agents)
    Agent(; policy, trajectory=CircularArrayTrajectory(;
        capacity=trajectory_length,
        state=Float32 => (ns, n_agents), action=Float32 => (na, n_agents),
        reward=Float32 => (n_agents,), terminated=Bool => (n_agents,),
        truncated=Bool => (n_agents,), next_state=Float32 => (ns, n_agents),
        episode_start=Bool => (n_agents,)))
end

"""Reset inference memory at an environment reset; does not clear replay or counters."""
function reset_sac_lstm!(p::SACLSTMPolicy; batch_size=1)
    p.actor_internal_state = nothing
    p.prev_action = send_to_device(device(p.actor), zeros(Float32, size(p.action_space)[1], batch_size))
    p.prev_reward = send_to_device(device(p.actor), zeros(Float32, 1, batch_size))
    p.last_action = nothing
    p.episode_start_pending = true
    return p
end

"""Choose one action and advance actor memory. Call observe_sac_lstm! after env.step."""
function act_sac_lstm!(p::SACLSTMPolicy, obs; deterministic=false)
    obs = send_to_device(device(p.actor), reshape(Float32.(obs), size(p.state_space)[1], :))
    size(obs, 2) == size(p.prev_action, 2) || throw(DimensionMismatch("reset policy memory for this environment batch size"))
    hidden, p.actor_internal_state = _sac_lstm_step(p.actor.encoder, obs,
        p.prev_action, p.prev_reward, p.actor_internal_state)
    action, _ = _sac_lstm_draw(p.actor.head, p.rng,
        vcat(hidden, p.actor.shortcut(obs)); deterministic)
    p.last_action = send_to_host(action)
    return p.last_action
end

"""Provide the executed agent-space action and actual reward for the next decision."""
function observe_sac_lstm!(p::SACLSTMPolicy, action, rewards)
    b = size(p.prev_action, 2)
    p.prev_action = send_to_device(device(p.actor), reshape(Float32.(action), size(p.action_space)[1], b))
    r = rewards isa Number ? fill(Float32(rewards), 1, b) : reshape(Float32.(rewards), 1, b)
    p.prev_reward = send_to_device(device(p.actor), r)
    return p
end

function (p::SACLSTMPolicy)(env; deterministic=false)
    # Advance memory even when a warm-up policy supplies the executed action.
    action = act_sac_lstm!(p, state(env); deterministic)
    if !deterministic && p.update_step <= p.start_steps
        isnothing(p.start_policy) && throw(ArgumentError("start_steps requires start_policy"))
        action = p.start_policy(env)
        action = action isa AbstractVector{<:AbstractArray} ? reduce(hcat, action) :
                 reshape(action, size(p.action_space)[1], :)
        p.last_action = Float32.(action)
    end
    return p.last_action
end

function update!(p::SACLSTMPolicy, ::AbstractTrajectory, env::AbstractEnv, ::PreEpisodeStage)
    reset_sac_lstm!(p; batch_size=length(state(env)) ÷ size(p.state_space)[1])
end

function update!(t::AbstractTrajectory, p::SACLSTMPolicy, env::AbstractEnv, ::PreActStage, action)
    p.update_step += 1
    p.last_action = copy(action)
    push!(t; state=state(env), action=action,
          episode_start=fill(p.episode_start_pending, size(p.prev_action, 2)))
    p.episode_start_pending = false
end

function update!(t::AbstractTrajectory, p::SACLSTMPolicy, env::AbstractEnv, ::PostActStage)
    b = size(p.prev_action, 2)
    rewards = reward(env)
    push!(t[:reward], rewards isa Number ? fill(Float32(rewards), b) : vec(rewards))
    push!(t[:terminated], is_terminated(env))
    push!(t[:truncated], is_truncated(env))
    push!(t[:next_state], state(env))
    observe_sac_lstm!(p, p.last_action, rewards)
end

function update!(p::SACLSTMPolicy, t::AbstractTrajectory, ::AbstractEnv, ::PostActStage)
    length(t) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    _update!(p, t)
end

function _update!(p::SACLSTMPolicy, t::AbstractTrajectory; bc::Bool=false)
    for _ in 1:p.update_loops
        update!(p, sample_sac_lstm(p, t); bc)
    end
    return nothing
end
update_IL(p::SACLSTMPolicy, t::AbstractTrajectory) = _update!(p, t; bc=!isnothing(p.bc_alpha))

_sac_lstm_frame(x, lane, index) = ndims(x) == 3 ? view(x, :, lane, index) : view(x, :, index)
_sac_lstm_scalar(x, lane, index) = ndims(x) == 1 ? x[index] : x[lane, index]

_sac_lstm_flags(x, n_lanes, n) = reshape(collect(x) .!= 0, n_lanes, n)

"""
    sac_lstm_episode_starts(t)

All stored `(index, lane)` pairs at which an episode begins: a set
`episode_start` flag, or the transition after a terminated/truncated one.
Without an `episode_start` field (external SARTTS data) index 1 also counts.
With the flag, an unflagged index 1 is a wrapped-around episode remainder and
is excluded.
"""
function sac_lstm_episode_starts(t::AbstractTrajectory)
    n = length(t)
    n_lanes = ndims(t[:state]) == 3 ? size(t[:state], 2) : 1
    has_flag = haskey(t, :episode_start)
    begins = has_flag ? _sac_lstm_flags(t[:episode_start], n_lanes, n) : falses(n_lanes, n)
    has_flag || (begins[:, 1] .= true)
    if n > 1
        ended = _sac_lstm_flags(t[:terminated], n_lanes, n) .|
                _sac_lstm_flags(t[:truncated], n_lanes, n)
        begins[:, 2:end] .|= ended[:, 1:end-1]
    end
    return [(c[2], c[1]) for c in findall(begins)]
end

"""
Sample ordered windows, right-padded with zeros and a loss mask. Starts are
uniform over stored transitions (including short episode suffixes), or, with
`p.episode_starts_only`, uniform over stored episode beginnings. A window
never crosses termination, truncation, an explicit reset, or the buffer end.
External SAC trajectories need only the original six SARTTS fields; their
terminated/truncated flags must mark episode boundaries. Optional `starts`
and `lanes` allow reproducible inspection of a specific batch.
"""
function sample_sac_lstm(p::SACLSTMPolicy, t::AbstractTrajectory; starts=nothing, lanes=nothing)
    all(k -> haskey(t, k), SARTTS) || throw(ArgumentError("ordered SARTTS trajectory required"))
    n = length(t)
    n > 0 || throw(ArgumentError("cannot sample an empty trajectory"))
    all(k -> size(t[k], ndims(t[k])) == n, SARTTS) || throw(ArgumentError("trajectory contains an incomplete transition"))
    ns, na, len = size(p.state_space)[1], size(p.action_space)[1], p.sequence_length
    size(t[:state], 1) == ns && size(t[:next_state], 1) == ns && size(t[:action], 1) == na ||
        throw(DimensionMismatch("trajectory and policy dimensions differ"))
    n_lanes = ndims(t[:state]) == 3 ? size(t[:state], 2) : 1
    if isnothing(starts) && p.episode_starts_only
        candidates = sac_lstm_episode_starts(t)
        isempty(candidates) && throw(ArgumentError("trajectory contains no episode start"))
        picks = rand(p.rng, candidates, p.batch_size)
        starts, lanes = first.(picks), last.(picks)
    end
    starts = isnothing(starts) ? rand(p.rng, 1:n, p.batch_size) : starts
    b = length(starts)
    b > 0 || throw(ArgumentError("empty batch"))
    lanes = isnothing(lanes) ? rand(p.rng, 1:n_lanes, b) : lanes
    length(lanes) == b || throw(DimensionMismatch("starts and lanes must have equal lengths"))
    obs = zeros(Float32, ns, b, len + 1)
    prev_actions = zeros(Float32, na, b, len + 1)
    prev_rewards = zeros(Float32, 1, b, len + 1)
    actions = zeros(Float32, na, b, len)
    rewards, terminated, mask = (zeros(Float32, 1, b, len) for _ in 1:3)
    for j in 1:b
        index, lane = starts[j], lanes[j]
        1 <= index <= n && 1 <= lane <= n_lanes || throw(BoundsError())
        obs[:, j, 1] .= _sac_lstm_frame(t[:state], lane, index)
        for step in 1:len
            actions[:, j, step] .= _sac_lstm_frame(t[:action], lane, index)
            rewards[1, j, step] = _sac_lstm_scalar(t[:reward], lane, index)
            terminated[1, j, step] = _sac_lstm_scalar(t[:terminated], lane, index)
            obs[:, j, step + 1] .= _sac_lstm_frame(t[:next_state], lane, index)
            prev_actions[:, j, step + 1] .= actions[:, j, step]
            prev_rewards[1, j, step + 1] = rewards[1, j, step]
            mask[1, j, step] = 1f0
            boundary = terminated[1, j, step] != 0 ||
                _sac_lstm_scalar(t[:truncated], lane, index) || index == n ||
                (haskey(t, :episode_start) && _sac_lstm_scalar(t[:episode_start], lane, index + 1))
            boundary && break
            index += 1
        end
    end
    return (; obs, prev_actions, prev_rewards, actions, rewards, terminated, mask)
end

_sac_lstm_mean(x, mask) = sum(x .* mask) / max(sum(mask), 1f0)

function _sac_lstm_target(p::SACLSTMPolicy, batch)
    next_actions, next_log_pi = sac_lstm_actions(p.actor, batch.obs,
        batch.prev_actions, batch.prev_rewards, p.rng)
    q1, q2 = sac_lstm_qvalues(p.target_critic, batch.obs,
        batch.prev_actions, batch.prev_rewards, next_actions)
    soft_values = min.(q1, q2) .- p.α .* next_log_pi
    targets = batch.rewards .+ p.γ .* (1f0 .- batch.terminated) .* soft_values[:, :, 2:end]
    return targets, next_log_pi[:, :, 2:end]
end

# `bc = true` only from IL updates: Q-filtered BC term (MSE of the mean action
# tanh(μ) to the expert action, masked) with the TD3+BC weighting of the actor objective.
function update!(p::SACLSTMPolicy, batch::NamedTuple{(:obs, :prev_actions, :prev_rewards, :actions, :rewards, :terminated, :mask)}; bc::Bool=false)
    batch = send_to_device(device(p.actor), batch)
    mask = batch.mask
    sum(mask) > 0 || throw(ArgumentError("batch has no valid transitions"))
    targets, next_log_pi = _sac_lstm_target(p, batch)
    p.last_target_q_mean = _sac_lstm_mean(targets, mask)
    p.last_mean_minus_log_pi = _sac_lstm_mean(-next_log_pi, mask)
    if isnothing(p.actor_state_tree) || isnothing(p.critic_state_tree)
        p.actor_state_tree = Flux.setup(p.optimizer_actor, p.actor)
        p.critic_state_tree = Flux.setup(p.optimizer_critic, p.critic)
        p.log_α_state_tree = Flux.setup(p.optimizer_log_α, p.log_α)
    end
    q_grad = Flux.gradient(p.critic) do critic
        q1, q2 = sac_lstm_qvalues(critic, batch.obs, batch.prev_actions,
                                batch.prev_rewards, batch.actions)
        loss1, loss2 = _sac_lstm_mean((q1 .- targets).^2, mask), _sac_lstm_mean((q2 .- targets).^2, mask)
        ignore_derivatives() do
            p.last_q1_mean, p.last_q2_mean = _sac_lstm_mean(q1, mask), _sac_lstm_mean(q2, mask)
            p.last_critic1_loss, p.last_critic2_loss = loss1, loss2
        end
        loss1 + loss2
    end
    Flux.update!(p.critic_state_tree, p.critic, q_grad[1])
    if p.use_popart
        valid_targets = vec(send_to_host(targets))[vec(send_to_host(mask)) .> 0]
        update!(p.critic.qnetwork1[end], valid_targets)
        update!(p.critic.qnetwork2[end], valid_targets)
    end
    # Critic history is fixed when varying the current action. In particular,
    # sampled policy actions never replace the recorded history in the LSTM.
    critic_hidden = sac_lstm_history(p.critic.encoder, batch.obs,
                                     batch.prev_actions, batch.prev_rewards)[:, :, 1:end-1]
    obs = batch.obs[:, :, 1:end-1]
    actor_grad = Flux.gradient(p.actor) do actor
        actions, log_pi = sac_lstm_actions(actor, batch.obs, batch.prev_actions,
                                           batch.prev_rewards, p.rng)
        q1, q2 = _sac_lstm_qheads(p.critic, critic_hidden, obs, actions[:, :, 1:end-1])
        value = _sac_lstm_mean(min.(q1, q2), mask)
        entropy_term = p.α * _sac_lstm_mean(log_pi[:, :, 1:end-1], mask)
        ignore_derivatives() do
            p.last_reward_term, p.last_entropy_term = value, entropy_term
            p.last_actor_loss = entropy_term - value
        end
        if bc
            λ = ignore_derivatives() do
                p.bc_alpha / (_sac_lstm_mean(abs.(min.(q1, q2)), mask) + 1f-6)
            end
            mean_actions, _ = sac_lstm_actions(actor, batch.obs, batch.prev_actions,
                                               batch.prev_rewards, p.rng; deterministic=true)
            a_mean = mean_actions[:, :, 1:end-1]
            # Q-filter (Nair et al. 2018) against the policy's mean action, with the
            # same critic history; rejected and padded steps count 0.
            accept = ignore_derivatives() do
                qe1, qe2 = _sac_lstm_qheads(p.critic, critic_hidden, obs, batch.actions)
                qm1, qm2 = _sac_lstm_qheads(p.critic, critic_hidden, obs, a_mean)
                Float32.(min.(qe1, qe2) .> min.(qm1, qm2))
            end
            bc_loss = _sac_lstm_mean(accept .* mean((a_mean .- batch.actions) .^ 2; dims=1), mask)
            ignore_derivatives() do
                p.last_bc_loss = bc_loss
                p.last_bc_accept = _sac_lstm_mean(accept, mask)
            end
            λ * (entropy_term - value) + bc_loss
        else
            entropy_term - value
        end
    end
    Flux.update!(p.actor_state_tree, p.actor, actor_grad[1])
    if p.automatic_entropy_tuning
        # Same next-action entropy objective and log-alpha clamp as agent_sac.jl.
        entropy_error = p.last_mean_minus_log_pi - p.target_entropy
        grad = Flux.gradient(p.log_α) do log_alpha
            exp(log_alpha[1]) * entropy_error
        end
        Flux.update!(p.log_α_state_tree, p.log_α, grad[1])
        clamp!(p.log_α, -12.5f0, 1.5f0)
        p.α = exp(p.log_α[1])
    end
    for (dest, src) in zip(Flux.trainables(p.target_critic), Flux.trainables(p.critic))
        dest .= (1f0 - p.τ) .* dest .+ p.τ .* src
    end
    if p.use_popart
        for (target, current) in ((p.target_critic.qnetwork1[end], p.critic.qnetwork1[end]),
                                  (p.target_critic.qnetwork2[end], p.critic.qnetwork2[end]))
            target.μ = (1f0 - p.τ) * target.μ + p.τ * current.μ
            target.σ = (1f0 - p.τ) * target.σ + p.τ * current.σ
        end
    end
    p.gradient_step += 1
    if p.verbose && p.gradient_step % 100 == 0
        println("SAC-LSTM update ", p.gradient_step, ": actor=", p.last_actor_loss,
                " critic=", p.last_critic1_loss + p.last_critic2_loss, " alpha=", p.α)
    end
    return nothing
end
