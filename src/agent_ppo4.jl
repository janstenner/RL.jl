
Base.@kwdef mutable struct PPOPolicy4 <: AbstractPolicy
    approximator
    trajectory_no_explore
    exploration_mode::Bool = false
    exploration_triggered::Bool = false
    γ::Float32 = 0.99f0
    λ::Float32 = 0.95f0
    clip_range::Float32 = 0.2f0
    n_microbatches::Int = 4
    n_epochs::Int = 4
    actor_loss_weight::Float32 = 1.0f0
    critic_loss_weight::Float32 = 0.5f0
    entropy_loss_weight::Float32 = 0.01f0
    rng = Random.GLOBAL_RNG
    update_freq::Int = 256
    update_freq_no_explore::Int = 256
    update_step::Int = 0
    update_step_no_explore::Int = 0
    n_updates::Int = 0
    clip1::Bool = false
    normalize_advantage::Bool = false
    start_steps = -1
    start_policy = nothing
    target_kl = 100.0
    fear_factor = 1.0f0
    fear_scale = 0.5f0
    new_loss = true
    last_action_log_prob::Vector{Float32} = [0.0f0]
    last_sigma::Vector{Float32} = [0.0f0]
    last_mu::Vector{Float32} = [0.0f0]
    deterministic::Bool = true
    trigger_set = nothing
end


Base.@kwdef mutable struct ActorCritic4{A,C,C2}
    actor::A
    actor_explore::A
    critic_no_explore::C # value function approximator
    critic::C2 # value function approximator for exploration
    critic_no_explore_frozen::C
    optimizer_actor = ADAM()
    optimizer_critic_no_explore = ADAM()
    optimizer_critic = ADAM()
    actor_state_tree = nothing
    critic_no_explore_state_tree = nothing
    critic_state_tree = nothing
end

@forward ActorCritic4.critic device


function perturb_sparse(model, σ, p_mask)
    for p in Flux.params(model)
        # skip biases
        if length(size(p)) > 1
            mask = rand(size(p)...) .< p_mask
            p .+= mask .* (σ * randn(Float32, size(p)))
        end
    end

    model
end


function create_agent_ppo4(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = relu, fun_critic = nothing, tanh_end = false, n_actors = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = false, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, fear_factor = 1.0, fear_scale = 0.5, new_loss = true, update_freq_no_explore = 256)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    critic_no_explore = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic)

    actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                )
    
    

    Agent(
        policy = PPOPolicy4(
            approximator = isnothing(approximator) ? ActorCritic4(
                actor = actor,
                actor_explore = perturb_sparse(deepcopy(actor), 0.1f0, 0.1f0),
                critic_no_explore = critic_no_explore,
                critic = deepcopy(critic_no_explore),
                critic_no_explore_frozen = deepcopy(critic_no_explore),
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
                optimizer_critic_no_explore = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
            ) : approximator,
            trajectory_no_explore = CircularArrayTrajectory(;
                capacity = update_freq_no_explore,
                state = Float32 => (size(state_space)[1], n_actors),
                reward = Float32 => (n_actors,),
                terminal = Bool => (n_actors,),
                next_states = Float32 => (size(state_space)[1], n_actors),
            ),
            γ = y,
            λ = p,
            #exploration_triggered = falses(n_actors),
            clip_range = clip_range,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            rng = rng,
            update_freq = update_freq,
            update_freq_no_explore = update_freq_no_explore,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            fear_factor = fear_factor,
            fear_scale = fear_scale,
            new_loss = new_loss,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_actors),
                action = Float32 => (size(action_space)[1], n_actors),
                action_log_prob = Float32 => (n_actors,),
                reward = Float32 => (n_actors,),
                terminal = Bool => (n_actors,),
                exploration_triggered = Bool => (n_actors,),
                next_states = Float32 => (size(state_space)[1], n_actors),
        ),
    )
end





function prob(p::PPOPolicy4, state::AbstractArray, explore::Bool = false)
    if explore
        # use the exploration actor
        μ, logσ = p.approximator.actor_explore(send_to_device(device(p.approximator), state)) |> send_to_host
    else
        # use the normal actor
        μ, logσ = p.approximator.actor(send_to_device(device(p.approximator), state)) |> send_to_host
    end

    StructArray{Normal}((μ, exp.(logσ)))
end

function prob(p::PPOPolicy4, env::AbstractEnv, explore::Bool = false)
    prob(p, state(env), explore)
end


function (p::PPOPolicy4)(env::AbstractEnv; ignore_explore_mode = false)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else

        if (p.exploration_mode || ignore_explore_mode) && p.exploration_triggered
            if p.deterministic
                action = p.approximator.actor_explore.μ(state(env))
                μ = action
                σ = [0.0f0]
            else
                dist = prob(p, env, true)
                action = rand.(p.rng, dist)
                μ = dist.μ
                σ = dist.σ
            end

            if p.clip1
                clamp!(action, -1.0, 1.0)
            end
        else
            if p.deterministic
                action = p.approximator.actor.μ(state(env))
                μ = action
                σ = [0.0f0]
            else
                dist = prob(p, env, false)
                action = rand.(p.rng, dist)
                μ = dist.μ
                σ = dist.σ
            end
        end

        # put the last action log prob behind the clip
        if p.deterministic
            log_p = [0.0f0]
        else
            if ndims(action) == 2
                log_p = vec(sum(normlogpdf(dist.μ, dist.σ, action), dims=1))
            else
                log_p = normlogpdf(dist.μ, dist.σ, action)
            end
        end

        

        p.last_action_log_prob = log_p
        p.last_mu = μ[:]
        p.last_sigma = σ[:]

        action
    end
end





"""
    local_interval(v::AbstractVector{<:Real}, x::Real; mass::Float64=0.1)

Given samples `v` and a target point `x`, returns (low, high) = the
interval [low, high] containing fraction `mass` of the empirical
distribution of `v`, centered (in quantile‐space) on x.  If x is near
the edges, the interval will be truncated to [minimum(v), maximum(v)].
"""
function local_interval(v::AbstractVector{<:Real}, x::Real; mass=0.1)
    @assert 0.0 < mass <= 1.0 "mass must be in (0,1]"
    # Empirical CDF and its inverse via quantiles
    cdf = ecdf(v)
    qx = cdf(x)                      # quantile of x
    half = mass/2
    q_low  = max(0.0, qx - half)
    q_high = min(1.0, qx + half)
    lower  = quantile(v, q_low)
    upper  = quantile(v, q_high)
    return lower, upper
end

"""
Pick a random row `s_r` from `traj` and for each column i compute
an interval around `s_r[i]` containing fraction `mass` of the samples
in that dimension.

Returns `(s_r, intervals)` where
  • s_r      is the chosen 1×D row,
  • intervals is a 2×D matrix whose i‑th column is [low; high] for dim i.
"""
function compute_intervals(traj::AbstractTrajectory; mass=0.1)
    states = flatten_batch(traj[:state])
    D, N = size(states)
    idx = rand(1:N)
    s_r = states[:, idx]

    intervals = Array{Float32}(undef, 2, D)
    for i in 1:D
        v = states[i, :]
        intervals[:, i] .= local_interval(v, s_r[i]; mass=mass)
    end
    return intervals
end

is_inside(intervals::AbstractMatrix{<:Real}, s_x::AbstractVector{<:Real}) =
    all((intervals[1, :] .<= s_x) .& (s_x .<= intervals[2, :]))

is_inside(intervals::AbstractMatrix{<:Real}, s_matrix::AbstractMatrix{<:Real}) =
    any(
      all(
        (intervals[1, :] .<= s_matrix) .& (s_matrix .<= intervals[2, :]),
        dims = 1
      )
    )




function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy4,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    if policy.exploration_mode
        push!(
            trajectory;
            state = state(env),
            action = action,
            action_log_prob = policy.last_action_log_prob,
            exploration_triggered = policy.exploration_triggered,
        )
    else
        push!(
            policy.trajectory_no_explore;
            state = state(env),
        )
    end
end



function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy4,
    env::AbstractEnv,
    ::PostActStage,
)
    if policy.exploration_mode
        r = reward(env)[:]

        push!(trajectory[:reward], r)
        push!(trajectory[:terminal], is_terminated(env))
        push!(trajectory[:next_states], state(env))
        #push!(trajectory[:next_values], policy.approximator.critic(send_to_device(device(policy.approximator), env.state)) |> send_to_host)
    else
        r = reward(env)[:]

        push!(policy.trajectory_no_explore[:reward], r)
        push!(policy.trajectory_no_explore[:terminal], is_terminated(env))
        push!(policy.trajectory_no_explore[:next_states], state(env))
    end

    # check for exploration trigger
    if policy.exploration_mode && !policy.exploration_triggered
        if is_inside(policy.trigger_set, env.state)
            policy.exploration_triggered = true
        end
    end
end



function update!(
    p::PPOPolicy4,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    if p.exploration_mode

        p.update_step += 1

        if p.update_step % p.update_freq == 0
            p.n_updates += 1
            _update!(p, t)
            
            p.exploration_mode = false
            p.update_step_no_explore = 0
        end
    else
        p.update_step_no_explore += 1

        if p.update_step_no_explore % p.update_freq_no_explore == 0
            p.trigger_set = compute_intervals(p.trajectory_no_explore; mass=0.8)
            p.approximator.actor_explore = perturb_sparse(deepcopy(p.approximator.actor), 0.06f0, 0.8f0)
            p.exploration_mode = true
            p.update_step = 0
        end
    end
    
end

function update!(
    p::PPOPolicy4,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    p.exploration_triggered = false
end





function nstep_targets(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    n::Int=3
) :: Vector{Float32}
    T = length(rewards)
    targets = similar(rewards)
    for t in 1:T
        g = 0f0
        discount = 1f0
        hit_done = false

        # k-Schritte von Belohnungen sammeln
        for k in 0:(n-1)
            idx = t + k
            if idx > T
                break
            end
            g += discount * rewards[idx]
            if dones[idx]        # Episode endet hier
                hit_done = true
                break
            end
            discount *= γ
        end

        # Nur bootstrappen, wenn in den ersten n Schritten kein Done war
        idxn = t + n - 1
        if !hit_done && idxn ≤ T && !dones[idxn]
            g += discount * next_values[idxn]
        end

        targets[t] = g
    end
    return targets
end


function td_lambda_targets(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    λ::Float32=0.7f0
) :: Vector{Float32}
    T = length(rewards)
    targets = similar(rewards)
    Gλ = 0f0
    for t in T:-1:1
        # für Schritt t ist next_values[t] = V(s_{t+1})
        Gλ = rewards[t] + γ * ((1f0-λ)*next_values[t] + λ*Gλ) * (1 - dones[t])
        targets[t] = Gλ
    end
    return targets
end

function td_lambda_targets(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = td_lambda_targets(rewards[i, :], dones[i, :], next_values[i, :], γ; λ=λ)
    end

    return results
end


function lambda_truncated_targets(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
    n::Int=3
) :: Vector{Float32}

    # berechne alle k‑Step‑Returns mit next_values
    Gs = [nstep_targets(rewards, dones, next_values, γ; n=k) for k in 1:n]

    T = length(rewards)
    targets = similar(rewards)
    for t in 1:T
        sum_part = 0f0
        for k in 1:(n-1)
            sum_part += (1f0-λ)*λ^(k-1) * Gs[k][t]
        end
        sum_part += λ^(n-1) * Gs[n][t]
        targets[t] = sum_part
    end
    return targets
end

function lambda_truncated_targets(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
    n::Int=3
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = lambda_truncated_targets(rewards[i, :], dones[i, :], next_values[i, :], γ; λ=λ, n=n)
    end

    return results
end





function set_up_state_trees!(AC)

    if isnothing(AC.actor_state_tree)
        AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor)
    end

    if isnothing(AC.critic_no_explore_state_tree)
        AC.critic_no_explore_state_tree = Flux.setup(AC.optimizer_critic_no_explore, AC.critic_no_explore)
    end

    if isnothing(AC.critic_state_tree)
        AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
    end
end




function _update!(p::PPOPolicy4, t::Any)

    println("TRAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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


    stop_actor_update = false

    actor_losses = Float32[]
    entropy_losses = Float32[]
    critic_no_explore_losses = Float32[]
    critic_losses = Float32[]
    

    excitements = Float32[]
    fears = Float32[]


    set_up_state_trees!(AC)
    


    if length(p.trajectory_no_explore) > 0
        tt = p.trajectory_no_explore

        n_actors_no_explore, n_rollout_no_explore = size(tt[:terminal])
        microbatch_size_no_explore = Int(floor(n_actors_no_explore * n_rollout_no_explore ÷ n_microbatches))
    
        
        states_no_explore = to_device(flatten_batch(tt[:state]))
        v_ref_no_explore = AC.critic_no_explore_frozen( states_no_explore )[:]

        for epoch in 1:n_epochs

            next_values_no_explore = reshape( AC.critic_no_explore( tt[:next_states] ), n_actors_no_explore, :)
            targets_no_explore = lambda_truncated_targets(tt[:reward], tt[:terminal], next_values_no_explore, γ)[:]
            
            rand_inds = shuffle!(rng, Vector(1:n_actors_no_explore*n_rollout_no_explore))
            for i in 1:n_microbatches
                batch_inds = rand_inds[(i-1)*microbatch_size_no_explore+1:i*microbatch_size_no_explore]

                s = select_last_dim(states_no_explore, batch_inds)

                g_critic_no_explore = Flux.gradient(AC.critic_no_explore) do critic_no_explore
                    v = critic_no_explore( s )[:]

                    bellman = mean((targets_no_explore[batch_inds] .- v) .^ 2)
                    fr_term = mean((v .- v_ref_no_explore[batch_inds]) .^ 2)
                    critic_no_explore_loss = bellman + 0.1 * fr_term
                    loss = w₂ * critic_no_explore_loss
                    ignore() do
                        push!(critic_no_explore_losses, w₂ * critic_no_explore_loss)
                    end

                    loss
                end
                
                Flux.update!(AC.critic_no_explore_state_tree, AC.critic_no_explore, g_critic_no_explore[1])
            end
        end

    end


    n_actors, n_rollout = size(t[:terminal])
    
    microbatch_size = Int(floor(n_actors * n_rollout ÷ n_microbatches))

    states = to_device(flatten_batch(t[:state]))
    next_states = to_device(flatten_batch(t[:next_states]))
    actions = to_device(flatten_batch(t[:action]))
    rewards = to_device(t[:reward])
    terminal = to_device(t[:terminal])
    action_log_probs = to_device(t[:action_log_prob])
    exploration_triggered = to_device(t[:exploration_triggered])



    AC.critic = deepcopy(AC.critic_no_explore)

    v_ref = AC.critic( states )[:] # no critic frozen here

    for epoch in 1:n_epochs

        next_values = reshape( AC.critic( next_states ), n_actors, :)
        targets = lambda_truncated_targets(rewards, terminal, next_values, γ)[:]

        rand_inds = shuffle!(rng, Vector(1:n_actors*n_rollout))
        for i in 1:n_microbatches
            batch_inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            s = select_last_dim(states, batch_inds)

            g_critic = Flux.gradient(AC.critic) do critic
                v = critic( s )[:]

                bellman = mean((targets[batch_inds] .- v) .^ 2)
                fr_term = mean((v .- v_ref[batch_inds]) .^ 2)
                critic_loss = bellman + 0.1 * fr_term
                loss = w₂ * critic_loss
                ignore() do
                    push!(critic_losses, w₂ * critic_loss)
                end

                loss
            end
            
            Flux.update!(AC.critic_state_tree, AC.critic, g_critic[1])
        end
    end


    # compute advantages
    advantages = zeros(Float32, n_actors, n_rollout)
    temp_exploration_triggered = false
    temp_advantages = zeros(Float32, n_actors)

    for j in 1:n_rollout
        if temp_exploration_triggered

            for i in 1:n_actors
                advantages[i,j] = temp_advantages[i]
            end

            if terminal[1,j]
                temp_exploration_triggered = false
            end
        else
            if exploration_triggered[1,j]
                temp_exploration_triggered = true
                temp_advantages = AC.critic(t[:state][:,:,j]) .- AC.critic_no_explore(t[:state][:,:,j])
            end
        end
    end
    

    for epoch in 1:n_epochs
        rand_inds = shuffle!(rng, Vector(1:n_actors*n_rollout))
        for i in 1:n_microbatches
            batch_inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            s = select_last_dim(states, batch_inds)
            a = select_last_dim(actions, batch_inds)

            log_p = vec(action_log_probs)[batch_inds]
            adv = vec(advantages)[batch_inds]



            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            

            g_actor = Flux.gradient(AC.actor) do actor

                if p.deterministic

                    a_pred = actor.μ(s)[:]

                    w = clamp.(exp.(adv), 0.0, 10.0)

                    entropy_loss = 0.0
                    fear = abs(minimum(adv))

                    actor_loss = sum( w .* (a_pred - a[:]).^2 )

                    loss = w₁ * actor_loss
                else

                    μ, logσ = actor(s)
                    σ = (exp.(logσ))

                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims=1))
                    else
                        log_p′ₐ = normlogpdf(μ, σ, a)
                    end

                    #clamp!(log_p′ₐ, log(1e-8), Inf)

                    # d = Anzahl der Aktionsdimensionen
                    d = size(logσ, 1)
                    # Konstante c = ½∙(ln(2π)+1)
                    c = 0.5f0 * (log(2π) + 1f0)
                    # Entropie pro Sample: sum(logσ, dims=1) + d*c
                    ent_per_sample = sum(logσ; dims=1) .+ d * c
                    # Mittlerer Entropie‑Loss (negiert, falls Sie maximieren wollen)
                    entropy_loss = mean(ent_per_sample)


                    eps = Float32(1e-8)
                    ratio = (exp.(log_p′ₐ) .+ eps) ./ (exp.(log_p) .+ eps) # avoid division by zero

                    ignore() do
                        # println(size(ratio))
                        # println(ratio)
                        # error("abgebrochen")
                        
                        if !stop_actor_update
                            approx_kl_div = mean(((ratio .- 1) - log.(ratio))  .* exp_m) |> send_to_host

                            if approx_kl_div > p.target_kl
                                println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                                stop_actor_update = true
                            end
                        end
                    end

                    fear = (abs.((ratio .- 1)) + ones(size(ratio))).^1.3 .* p.fear_factor

                    if p.new_loss
                        actor_loss = -mean(((ratio .* adv) - fear)) # .* exp_m[:])
                    else
                        surr1 = ratio .* adv
                        surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                        actor_loss = -mean(min.(surr1, surr2)[KL_idx] .* exp_m[KL_idx])
                    end

                    loss = w₁ * actor_loss - w₃ * entropy_loss
                end


                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)

                    push!(excitements, maximum(adv))
                    push!(fears, maximum(fear))
                end

                loss
            end
            
            
            if !stop_actor_update
                Flux.update!(AC.actor_state_tree, AC.actor, g_actor[1])
            else
                break
            end

        end

        if stop_actor_update
            break
        end

    end




    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic_no_explore_loss = mean(abs.(critic_no_explore_losses))
    mean_critic_loss = mean(abs.(critic_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    
    println("---")
    println("mean actor loss: $(mean_actor_loss)")
    println("mean critic_no_explore loss: $(mean_critic_no_explore_loss)")
    println("mean critic loss: $(mean_critic_loss)")
    println("mean entropy loss: $(mean_entropy_loss)")

    mean_excitement = mean(abs.(excitements))
    max_fear = maximum(abs.(fears))
    
    println("mean excitement: $(mean_excitement)")
    println("max fear: $(max_fear)")

    if p.new_loss
        new_factor_factor = mean_excitement * p.fear_scale

        println("changing fear factor from $(p.fear_factor) to $(p.fear_factor * 0.9 + new_factor_factor * 0.1)")

        p.fear_factor = p.fear_factor * 0.9 + new_factor_factor * 0.1
    end

    println("---")
end