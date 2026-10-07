# TD3 (Fujimoto et al. 2018, arXiv 1802.09477) with
#   - clipped double Q-learning (twin critics, min target),
#   - target policy smoothing (clipped Gaussian noise on the target action),
#   - delayed actor and target updates (`policy_delay`),
#   - optional LayerNorm in the critic (and actor) hidden layers,
#   - small uniform init of the last actor layer (Lillicrap et al. 2015),
#   - TD3+BC actor loss for imitation learning via `update_IL`
#     (Fujimoto & Gu 2021, arXiv 2106.06860), with the BC term restricted by
#     a Q-filter (Nair et al. 2018, arXiv 1709.10089).
#
# Differences to agent_ddpg.jl besides the algorithm itself:
#   - transitions are stored with an explicit `next_state` and the policy is
#     updated in PostActStage, so all traces are aligned when sampling (also
#     after the circular buffer wrapped around),
#   - rewards and done masks are flattened to vectors before computing the TD
#     target (no (1, B) .+ (B,) broadcasting to a B×B matrix).


function create_NNA_td3(; ns, na, is_actor, width, network_depth = 2, fun = relu,
                        init, layer_norm = false, final_init = nothing, use_gpu = false)
    network_depth = max(1, Int(network_depth))
    n_in = is_actor ? ns : ns + na

    layers = Any[]
    for i in 1:network_depth
        n_prev = i == 1 ? n_in : width
        if layer_norm
            # Dense -> LayerNorm -> activation (as in RLPD / BRO critics)
            push!(layers, Dense(n_prev, width; init = init))
            push!(layers, LayerNorm(width, fun))
        else
            push!(layers, Dense(n_prev, width, fun; init = init))
        end
    end

    last_init = isnothing(final_init) ? init : final_init
    if is_actor
        push!(layers, Dense(width, na, tanh; init = last_init))
    else
        push!(layers, Dense(width, 1; init = last_init))
    end

    n = Chain(layers...)
    use_gpu ? n |> gpu : n
end


function create_agent_td3(; action_space, state_space, use_gpu = false, rng,
                    y = 0.99f0, p = 0.995f0, batch_size = 256,
                    start_steps = 10_000, update_after = 10_000, update_freq = 1, update_loops = 1,
                    policy_delay = 2, target_noise = 0.2f0, target_noise_clip = 0.5f0,
                    act_limit = 1.0, act_noise = 0.1, noise_hold = 1,
                    nna_scale = 1, nna_scale_critic = nothing,
                    network_depth = 2, network_depth_critic = nothing,
                    fun = relu, fun_critic = nothing,
                    critic_layer_norm = true, actor_layer_norm = false,
                    actor_final_init_scale = 3f-3,
                    bc_alpha = 2.5f0,
                    trajectory_length = 1_000_000,
                    learning_rate = 1e-4, learning_rate_critic = nothing,
                    clip_grad = 1.0, betas = (0.9, 0.999),
                    reset_stage = nothing, verbose = false)

    isnothing(nna_scale_critic)     && (nna_scale_critic = nna_scale)
    isnothing(network_depth_critic) && (network_depth_critic = network_depth)
    isnothing(fun_critic)           && (fun_critic = fun)
    isnothing(learning_rate_critic) && (learning_rate_critic = learning_rate)

    # same width convention as agent_ddpg.jl
    width_actor = Int(floor(10 * nna_scale))
    width_critic = Int(floor(20 * nna_scale_critic))

    ns = size(state_space)[1]
    na = size(action_space)[1]

    init = Flux.glorot_uniform(rng)
    final_init = isnothing(actor_final_init_scale) ? nothing :
        (dims...) -> Float32(actor_final_init_scale) .* (2 .* rand(rng, Float32, dims...) .- 1)

    behavior_actor = create_NNA_td3(ns = ns, na = na, is_actor = true, width = width_actor,
        network_depth = network_depth, fun = fun, init = init,
        layer_norm = actor_layer_norm, final_init = final_init, use_gpu = use_gpu)

    behavior_critic1 = create_NNA_td3(ns = ns, na = na, is_actor = false, width = width_critic,
        network_depth = network_depth_critic, fun = fun_critic, init = init,
        layer_norm = critic_layer_norm, use_gpu = use_gpu)
    behavior_critic2 = create_NNA_td3(ns = ns, na = na, is_actor = false, width = width_critic,
        network_depth = network_depth_critic, fun = fun_critic, init = init,
        layer_norm = critic_layer_norm, use_gpu = use_gpu)

    Agent(
        policy = TD3Policy(
            action_space = action_space,
            state_space = state_space,
            rng = rng,

            behavior_actor = behavior_actor,
            behavior_critic1 = behavior_critic1,
            behavior_critic2 = behavior_critic2,
            target_actor = deepcopy(behavior_actor),
            target_critic1 = deepcopy(behavior_critic1),
            target_critic2 = deepcopy(behavior_critic2),

            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
            optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate_critic, betas)),

            use_gpu = use_gpu,
            y = y,
            p = p,
            batch_size = batch_size,
            start_steps = start_steps,
            update_after = update_after,
            update_freq = update_freq,
            update_loops = update_loops,
            policy_delay = policy_delay,
            target_noise = Float32(target_noise),
            target_noise_clip = Float32(target_noise_clip),
            act_limit = Float32(act_limit),
            act_noise = act_noise,
            noise_hold = noise_hold,
            last_noise = zeros(Float32, na, 1),
            bc_alpha = isnothing(bc_alpha) ? nothing : Float32(bc_alpha),
            reset_stage = reset_stage,
            verbose = verbose,
        ),
        trajectory =
            CircularArrayTrajectory(;
                capacity = trajectory_length,
                state = Float32 => ns,
                action = Float32 => na,
                reward = Float32 => (),
                terminated = Bool => (),
                truncated = Bool => (),
                next_state = Float32 => ns,
            ),
    )
end


Base.@kwdef mutable struct TD3Policy{R} <: AbstractPolicy

    action_space::Space
    state_space::Space

    rng::R

    # `behavior_actor` keeps the DDPG name: validation, render_run and the
    # hook access the actor through it.
    behavior_actor
    behavior_critic1
    behavior_critic2
    target_actor
    target_critic1
    target_critic2

    optimizer_actor
    optimizer_critic
    actor_state_tree = nothing
    critic1_state_tree = nothing
    critic2_state_tree = nothing

    use_gpu::Bool

    y
    p
    batch_size
    start_steps
    update_after
    update_freq
    update_loops
    policy_delay::Int = 2
    target_noise::Float32 = 0.2f0
    target_noise_clip::Float32 = 0.5f0
    act_limit
    act_noise
    noise_hold
    last_noise
    # TD3+BC weight for `update_IL`; `nothing` = plain TD3 updates on expert data
    bc_alpha::Union{Nothing, Float32} = 2.5f0
    reset_stage
    verbose::Bool = false

    update_step::Int = 0
    critic_updates::Int = 0

    # diagnostics of the last update
    last_actor_loss::Float32 = 0.0f0
    last_critic1_loss::Float32 = 0.0f0
    last_critic2_loss::Float32 = 0.0f0
    last_q1_mean::Float32 = 0.0f0
    last_target_q_mean::Float32 = 0.0f0
    last_bc_loss::Float32 = 0.0f0
    # share of expert samples accepted by the Q-filter in the last BC update
    last_bc_accept::Float32 = 0.0f0
    # fraction of actor outputs in the batch with |a| > 0.99 (tanh saturation)
    last_action_saturation::Float32 = 0.0f0
end


function (policy::TD3Policy)(env; learning = true)
    s = state(env)

    if policy.update_step <= policy.start_steps
        # uniform random warm-up actions
        return policy.act_limit .* (2 .* rand(policy.rng, Float32, size(policy.last_noise, 1), size(s, 2)) .- 1)
    end

    D = device(policy.behavior_actor)
    actions = policy.behavior_actor(send_to_device(D, s)) |> send_to_host

    if learning
        if policy.update_step % policy.noise_hold == 0 || size(policy.last_noise) != size(actions)
            policy.last_noise = randn(policy.rng, Float32, size(actions)) .* Float32(policy.act_noise)
        end
        actions = actions .+ policy.last_noise
    end

    clamp.(actions, -policy.act_limit, policy.act_limit)
end


function (policy::TD3Policy)(stage::AbstractStage, env::AbstractEnv)
    nothing
end


function update!(policy::TD3Policy, ::Trajectory, ::AbstractEnv, stage::Union{PostEpisodeStage, PostExperimentStage})
    if stage == policy.reset_stage
        policy.update_step = 0
    end
end

function update!(::AbstractTrajectory, ::TD3Policy, ::AbstractEnv, ::PreEpisodeStage)
end

function update!(::AbstractTrajectory, ::TD3Policy, ::AbstractEnv, ::PostEpisodeStage)
end

function update!(trajectory::AbstractTrajectory, policy::TD3Policy, env::AbstractEnv, ::PreActStage, action)
    policy.update_step += 1

    push!(trajectory[:state], vec(state(env)))
    push!(trajectory[:action], vec(action))
end

function update!(trajectory::AbstractTrajectory, policy::TD3Policy, env::AbstractEnv, ::PostActStage)
    push!(trajectory[:reward], Float32(reward(env)[1]))
    # the done mask is the true termination; truncation is bootstrapped
    push!(trajectory[:terminated], is_terminated(env))
    push!(trajectory[:truncated], is_truncated(env))
    push!(trajectory[:next_state], vec(state(env)))
end

# Updating after the PostActStage push keeps all traces aligned.
function update!(policy::TD3Policy, traj::Trajectory, ::AbstractEnv, ::PostActStage)
    length(traj) > policy.update_after || return
    policy.update_step % policy.update_freq == 0 || return

    for i = 1:policy.update_loops
        update!(policy, td3_sample(policy.rng, traj, policy.batch_size))
    end
end


"""
    td3_sample(rng, t, batch_size)

Samples `(s, a, r, d, s′)` with `r` and `d` as vectors. Trajectories with a
`next_state` trace (online TD3 buffer) are sampled directly; trajectories
without one (the DDPG expert data used for IL) take the next row of `state`,
which is only valid because those are written in one pass and never wrap.
"""
function td3_sample(rng::AbstractRNG, t::AbstractTrajectory, batch_size::Int)
    if haskey(t, :next_state)
        inds = rand(rng, 1:length(t), batch_size)
        s′ = Array(consecutive_view(t[:next_state], inds))
    else
        inds = rand(rng, 1:length(t)-1, batch_size)
        s′ = Array(consecutive_view(t[:state], inds .+ 1))
    end

    s = Array(consecutive_view(t[:state], inds))
    a = Array(consecutive_view(t[:action], inds))
    r = Float32.(vec(Array(consecutive_view(t[:reward], inds))))
    d = Float32.(vec(Array(consecutive_view(t[:terminated], inds))))

    (s = s, a = a, r = r, d = d, s′ = s′)
end


# imitation learning: TD3 critic updates plus the TD3+BC actor loss
function update_IL(p::TD3Policy, t::AbstractTrajectory)
    for i = 1:p.update_loops
        update!(p, td3_sample(p.rng, t, p.batch_size); bc = !isnothing(p.bc_alpha))
    end
end


function td3_soft_update!(target, source, p)
    for (dest, src) in zip(Flux.trainables(target), Flux.trainables(source))
        dest .= p .* dest .+ (1 - p) .* src
    end
end


function update!(policy::TD3Policy, batch::NamedTuple; bc::Bool = false)

    if isnothing(policy.actor_state_tree)
        if policy.verbose
            println("________________________________________________________________________")
            println("Reset Optimizers")
            println("________________________________________________________________________")
        end
        policy.actor_state_tree = Flux.setup(policy.optimizer_actor, policy.behavior_actor)
        policy.critic1_state_tree = Flux.setup(policy.optimizer_critic, policy.behavior_critic1)
        policy.critic2_state_tree = Flux.setup(policy.optimizer_critic, policy.behavior_critic2)
    end

    A = policy.behavior_actor
    C1 = policy.behavior_critic1
    C2 = policy.behavior_critic2

    D = device(A)
    s, a, r, d, s′ = send_to_device(D, (batch.s, batch.a, batch.r, batch.d, batch.s′))

    # target policy smoothing
    ε = clamp.(policy.target_noise .* randn(policy.rng, Float32, size(a)),
               -policy.target_noise_clip, policy.target_noise_clip)
    a′ = clamp.(policy.target_actor(s′) .+ send_to_device(D, ε), -policy.act_limit, policy.act_limit)

    # clipped double Q target
    q′_input = vcat(s′, a′)
    q′ = min.(vec(policy.target_critic1(q′_input)), vec(policy.target_critic2(q′_input)))
    y = r .+ policy.y .* (1 .- d) .* q′
    policy.last_target_q_mean = mean(y)

    q_input = vcat(s, a)

    grad1 = Flux.gradient(C1) do critic
        q = vec(critic(q_input))
        loss = mean((y .- q) .^ 2)
        ignore_derivatives() do
            policy.last_critic1_loss = loss
            policy.last_q1_mean = mean(q)
        end
        loss
    end
    Flux.update!(policy.critic1_state_tree, C1, grad1[1])

    grad2 = Flux.gradient(C2) do critic
        q = vec(critic(q_input))
        loss = mean((y .- q) .^ 2)
        ignore_derivatives() do
            policy.last_critic2_loss = loss
        end
        loss
    end
    Flux.update!(policy.critic2_state_tree, C2, grad2[1])

    policy.critic_updates += 1
    policy.critic_updates % policy.policy_delay == 0 || return

    # delayed actor update
    actor_grad = Flux.gradient(A) do actor
        π_s = actor(s)
        q = vec(C1(vcat(s, π_s)))

        if bc
            # TD3+BC: λ = α / mean|Q| is treated as a constant
            λ = ignore_derivatives() do
                policy.bc_alpha / (mean(abs.(q)) + 1f-6)
            end
            # Q-filter (Nair et al. 2018): imitate only where the critic rates
            # the expert action above the policy's own; rejected samples count 0.
            accept = ignore_derivatives() do
                Float32.(vec(C1(vcat(s, a))) .> q)
            end
            bc_loss = mean(accept .* vec(mean((π_s .- a) .^ 2; dims = 1)))
            loss = -λ * mean(q) + bc_loss
            ignore_derivatives() do
                policy.last_bc_accept = mean(accept)
            end
        else
            bc_loss = 0.0f0
            loss = -mean(q)
        end

        ignore_derivatives() do
            policy.last_actor_loss = loss
            policy.last_bc_loss = bc_loss
            policy.last_action_saturation = mean(abs.(π_s) .> 0.99f0)
        end
        loss
    end
    Flux.update!(policy.actor_state_tree, A, actor_grad[1])

    # delayed target updates (polyak averaging)
    td3_soft_update!(policy.target_actor, A, policy.p)
    td3_soft_update!(policy.target_critic1, C1, policy.p)
    td3_soft_update!(policy.target_critic2, C2, policy.p)

    nothing
end
