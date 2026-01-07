
Base.@kwdef mutable struct SACPolicy2 <: AbstractPolicy
    actor::GaussianNetwork
    qnetwork1
    qnetwork2
    target_qnetwork1
    target_qnetwork2

    critic
    critic_frozen

    optimizer_actor = ADAM()
    optimizer_qnetwork1 = ADAM()
    optimizer_qnetwork2 = ADAM()
    optimizer_critic = ADAM()
    actor_state_tree = nothing
    qnetwork1_state_tree = nothing
    qnetwork2_state_tree = nothing
    critic_state_tree = nothing

    critic_frozen_factor

    action_space::Space
    state_space::Space

    γ::Float32 =0.99f0
    τ::Float32 =0.005f0
    α::Float32 =0.2f0

    log_α =[log(0.2f0)]
    optimizer_log_α = ADAM()
    log_α_state_tree = nothing


    batch_size::Int =32
    start_steps::Int =-1
    start_policy = nothing
    update_after::Int =1000
    update_freq::Int =50
    update_loops::Int = 1
    automatic_entropy_tuning::Bool =true
    lr_alpha::Float32 =0.0003f0
    target_entropy::Float32 =-1.0f0
    update_step::Int =0
    rng =Random.GLOBAL_RNG
    device_rng =Random.GLOBAL_RNG
    use_popart = false

    fear_factor = 0.1f0
    on_policy_update_freq = 2500
    λ_targets = 0.7f0
    target_frac = 0.3f0
    verbose::Bool = true

    antithetic_mean_samples::Int = 16
    on_policy_actor_loops::Int = 4

    # Logging
    last_reward_term::Float32 =0.0f0
    last_entropy_term::Float32 =0.0f0
    last_actor_loss::Float32 =0.0f0
    last_critic1_loss::Float32 =0.0f0
    last_critic2_loss::Float32 =0.0f0
    #last_log_alpha::Float32 =0.0f0
    last_q1_mean::Float32 =0.0f0
    last_q2_mean::Float32 =0.0f0
    last_target_q_mean::Float32 =0.0f0
    last_mean_minus_log_pi::Float32 =0.0f0
end


function create_agent_sac2(;action_space, state_space, use_gpu = false, rng, y, t =0.005f0, a =0.2f0, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, learning_rate = 0.00001, learning_rate_critic = nothing, fun = gelu, fun_critic = nothing, tanh_end = false, n_agents = 1, logσ_is_network = false, batch_size = 32, start_steps = -1, start_policy = nothing, update_after = 1000, update_freq = 50, update_loops = 1, max_σ = 7.0f0, min_σ = 2f-9, clip_grad = 0.5, start_logσ = 0.0, betas = (0.9, 0.999), trajectory_length = 10_000, automatic_entropy_tuning = true, lr_alpha = nothing, target_entropy = nothing, use_popart = false, critic_frozen_factor = 0.1f0, on_policy_update_freq = 2500, λ_targets= 0.7f0, fear_factor = 0.1f0, target_frac = 0.3f0, verbose = true, antithetic_mean_samples = 16, on_policy_actor_loops = 4)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(learning_rate_critic)     &&  (learning_rate_critic = learning_rate)

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    isnothing(target_entropy)           &&  (target_entropy = -Float32(na))

    isnothing(lr_alpha)                 && (lr_alpha = Float32(learning_rate))


    qnetwork1 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic, is_critic2 = true, popart = use_popart)
    target_qnetwork1 = deepcopy(qnetwork1)

    qnetwork2 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic, is_critic2 = true, popart = use_popart)
    target_qnetwork2 = deepcopy(qnetwork2)

    critic = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic, popart = use_popart)

    critic_frozen = deepcopy(critic)

    Agent(
        policy = SACPolicy2(
           actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ,
                    min_σ = min_σ,
                ),
            qnetwork1 = qnetwork1,
            qnetwork2 = qnetwork2,
            target_qnetwork1 = target_qnetwork1,
            target_qnetwork2 = target_qnetwork2,

            critic = critic,
            critic_frozen = critic_frozen,

            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
            optimizer_qnetwork1 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
            optimizer_qnetwork2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
            optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),

            critic_frozen_factor = critic_frozen_factor,

            action_space = action_space,
            state_space = state_space,
            
            γ = y,
            τ = t,
            α = a,
            log_α = [log(a)],
            optimizer_log_α = Optimisers.Adam(lr_alpha),

            batch_size = batch_size,
            start_steps = start_steps,
            start_policy = start_policy,
            update_after = update_after,
            update_freq = update_freq,
            update_loops = update_loops,
            automatic_entropy_tuning = automatic_entropy_tuning,
            lr_alpha = lr_alpha,
            target_entropy = target_entropy,
            rng = rng,
            device_rng = rng,
            use_popart = use_popart,

            fear_factor = fear_factor,
            on_policy_update_freq = on_policy_update_freq,
            λ_targets = λ_targets,
            target_frac = target_frac,
            verbose = verbose,

            antithetic_mean_samples = antithetic_mean_samples,
            on_policy_actor_loops = on_policy_actor_loops,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = trajectory_length,
                state = Float32 => (size(state_space)[1], n_agents),
                action = Float32 => (size(action_space)[1], n_agents),
                reward = Float32 => (n_agents),
                terminal = Bool => (n_agents,),
                #next_states = Float32 => (size(state_space)[1], n_agents),
        ),
    )
end



# TODO: handle Training/Testing mode
function (p::SACPolicy2)(env)

    if p.update_step <= p.start_steps
        action = p.start_policy(env)
        if(size(action[1]) != ())
            action = reduce(hcat, action)
        end
        action
    else
        # D = device(p.actor)
        # s = send_to_device(D, state(env))
        # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
        # trainmode:
        s = state(env)
        action = p.actor(p.device_rng, s; is_sampling=true)
        # action = dropdims(action, dims=ndims(action)) # Single action vec, drop second dim
        # send_to_host(action)

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "mean" action
        # action = dropdims(p.actor(s)[1], dims=2)

        action
    end
end








function update!(
    trajectory::AbstractTrajectory,
    policy::SACPolicy2,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    policy.update_step += 1
    
    push!(
        trajectory;
        state = state(env),
        action = action,
        #action_log_prob = policy.last_action_log_prob,
        #explore_mod = policy.mm.last,
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::SACPolicy2,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)[:]

    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
    #push!(trajectory[:next_states], state(env))
    #push!(trajectory[:next_values], policy.approximator.critic(send_to_device(device(policy.approximator), env.state)) |> send_to_host)
end

function update!(
    p::SACPolicy2,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    if length(size(p.action_space)) == 2
        number_actuators = size(p.action_space)[2]
    else
        #mono case 
        number_actuators = 1
    end

    length(t) > p.update_after * number_actuators || return

    
    
    if p.update_step % p.on_policy_update_freq == 0
        on_policy_update(p,t)
    end
    
    if p.update_step % p.update_freq == 0
        for i = 1:p.update_loops
            inds, batch = pde_sample(p.rng, t, BatchSampler{SARTS}(p.batch_size), number_actuators)
            update!(p, batch)
        end
    end
end





# ∥∇_a Q_min(s,a)∥ 
function grad_a_q_norms(qnet1, qnet2, s_batch, a_batch)
    qmin_sum(a_) = sum(min.(qnet1(vcat(s_batch, a_)), qnet2(vcat(s_batch, a_))))
    g = Zygote.gradient(qmin_sum, a_batch)[1]   # same shape as a_batch
    mags = sqrt.(sum(abs2, g; dims=1))[:]        # per-sample ‖∇_a Q_min‖
    mags
end


# ----- Q-basierte Maske für SAC: großes Q => kleine w -----
function q_rank_mask(q_batch::AbstractVector{<:Real};
                     target_frac::Float32=0.30f0, sharpness::Float32=20f0)
    q = Float32.(q_batch)
    B = length(q)
    r = (ordinalrank(q; rev=true) .- 1f0) ./ max(1f0, B-1f0)  # r=0 bei größtem Q
    @. 1f0 / (1f0 + exp(-sharpness*(r - target_frac))) |> Float32
end



# Anteil-Controller
function adjust_fear_factor(fear_factor, frac_changed; target=0.30f0, up=1.1f0, down=0.9f0, min = 1f-8, max = 1f8)
        
    new_factor = frac_changed > target ? fear_factor*up : fear_factor*down

    clamp(new_factor, min, max)
end


function check_state_trees(p::SACPolicy2)
    # just to make sure the state trees are initialized

    if isnothing(p.actor_state_tree) || isnothing(p.qnetwork1_state_tree) || isnothing(p.qnetwork2_state_tree)
        println("________________________________________________________________________")
        println("Reset Optimizers")
        println("________________________________________________________________________")
        p.actor_state_tree = Flux.setup(p.optimizer_actor, p.actor)
        p.qnetwork1_state_tree = Flux.setup(p.optimizer_qnetwork1, p.qnetwork1)
        p.qnetwork2_state_tree = Flux.setup(p.optimizer_qnetwork2, p.qnetwork2)

        p.critic_state_tree = Flux.setup(p.optimizer_critic, p.critic)

        Optimisers.adjust!(p.critic_state_tree.layers[end]; lambda = 0.0)
        Optimisers.adjust!(p.qnetwork1_state_tree.layers[end]; lambda = 0.0)
        Optimisers.adjust!(p.qnetwork2_state_tree.layers[end]; lambda = 0.0)

        p.log_α_state_tree = Flux.setup(p.optimizer_log_α, p.log_α)
    end

    return
end


function antithetic_mean_sac2(p, states, α; use_grad_a_q_norms = false, K = 16)
    μ, logσ = p.actor(p.device_rng, states)

    acc_mu  = zeros(Float32, 1, size(μ,2), size(μ,3)) 
    acc_logp = zeros(Float32, 1, size(μ,2), size(μ,3))
    acc_mags  = zeros(Float32, size(μ,2)* size(μ,3)) 

    if isdefined(p, :antithetic_mean_samples)
        K = p.antithetic_mean_samples
    end

    for k in 1:K
        a_plus, logp_π_plus, a_minus, logp_π_minus = p.actor(p.device_rng, states; is_sampling=true, is_return_log_prob=true, is_antithetic = true)


        if use_grad_a_q_norms

            acc_mags .+= grad_a_q_norms(p.qnetwork1, p.qnetwork2, states, a_plus)
            acc_mags .+= grad_a_q_norms(p.qnetwork1, p.qnetwork2, states, a_minus)

        else

            acc_logp .+= logp_π_plus .+ logp_π_minus


            y_plus1  = send_to_host(p.target_qnetwork1(vcat(states, a_plus)))
            y_minus1 = send_to_host(p.target_qnetwork1(vcat(states, a_minus)))

            y_plus2  = send_to_host(p.target_qnetwork2(vcat(states, a_plus)))
            y_minus2 = send_to_host(p.target_qnetwork2(vcat(states, a_minus)))


            acc_mu .+= min.(y_plus1, y_plus2) .- α .* logp_π_plus
            acc_mu .+= min.(y_minus1, y_minus2) .- α .* logp_π_minus
        end
    end


    if use_grad_a_q_norms
        return acc_mags
    else
        acc_mu ./= 2*K
        acc_logp ./= 2*K

        return acc_mu, acc_logp
    end
end



function on_policy_update(p::SACPolicy2, traj::AbstractTrajectory; whole_trajectory = false)

    check_state_trees(p)

    if !whole_trajectory
        n_samples = p.on_policy_update_freq
        s = send_to_device(device(p.qnetwork1), traj[:state][:,:,end-n_samples:end-1])
        a = send_to_device(device(p.qnetwork1), traj[:action][:,:,end-n_samples:end-1])
        r = send_to_device(device(p.qnetwork1), traj[:reward][:,end-n_samples:end-1])
        t = send_to_device(device(p.qnetwork1), traj[:terminal][:,end-n_samples:end-1])
        next_states = send_to_device(device(p.qnetwork1), traj[:state][:,:,end-n_samples+1:end])
    else
        n_samples = length(traj)
        s = send_to_device(device(p.qnetwork1), traj[:state])
        a = send_to_device(device(p.qnetwork1), traj[:action])
        r = send_to_device(device(p.qnetwork1), traj[:reward])
        t = send_to_device(device(p.qnetwork1), traj[:terminal])
        next_states = deepcopy(circshift(s, (0,0,-1)))
        #next_states[:,:, end] = zeros(Float32, size(s, 1), size(s, 2))  # terminal state
    end
    
    

    γ, τ, α = p.γ, p.τ, p.α


    acc_mu, acc_logp = antithetic_mean_sac2(p, next_states, α)
    
    logp_π′ = acc_logp

    next_values = acc_mu

    if whole_trajectory
        next_values[:,:, end] .*= 0.0f0     # terminal states
    end

    n_envs = size(t, 1)
    next_values = reshape( next_values, n_envs, :)
    
    targets = td_lambda_targets(r, t, next_values, γ; λ = p.λ_targets)

    q_input = vcat(s, a)

    q_grad_1 = Flux.gradient(p.qnetwork1) do qnetwork1
        q1 = dropdims(qnetwork1(q_input), dims=1)

        ignore_derivatives() do
            p.last_q1_mean = mean(q1)
            p.last_critic1_loss = Flux.mse(q1, targets)
        end

        Flux.mse(q1, targets)
    end
    Flux.update!(p.qnetwork1_state_tree, p.qnetwork1, q_grad_1[1])
    if p.use_popart
        update!(p.qnetwork1.layers[end], targets) 
    end

    q_grad_2 = Flux.gradient(p.qnetwork2) do qnetwork2
        q2 = dropdims(qnetwork2(q_input), dims=1)

        ignore_derivatives() do
            p.last_q2_mean = mean(q2)
            p.last_critic2_loss = Flux.mse(q2, targets)
        end

        Flux.mse(q2, targets)
    end
    Flux.update!(p.qnetwork2_state_tree, p.qnetwork2, q_grad_2[1])
    if p.use_popart
        update!(p.qnetwork2.layers[end], targets) 
    end


    # polyak averaging
    Functors.fmap(p.target_qnetwork1, p.qnetwork1) do tgt, src
        if tgt isa AbstractArray && src isa AbstractArray
            @. tgt = (1 - τ) * tgt + τ * src
        end
        tgt
    end

    Functors.fmap(p.target_qnetwork2, p.qnetwork2) do tgt, src
        if tgt isa AbstractArray && src isa AbstractArray
            @. tgt = (1 - τ) * tgt + τ * src
        end
        tgt
    end

    if p.use_popart
        # PopArt Polyak
        p.target_qnetwork1[end].μ = (1 - τ) .* p.target_qnetwork1[end].μ .+ τ .* p.qnetwork1[end].μ
        p.target_qnetwork1[end].σ = (1 - τ) .* p.target_qnetwork1[end].σ .+ τ .* p.qnetwork1[end].σ
        p.target_qnetwork2[end].μ = (1 - τ) .* p.target_qnetwork2[end].μ .+ τ .* p.qnetwork2[end].μ
        p.target_qnetwork2[end].σ = (1 - τ) .* p.target_qnetwork2[end].σ .+ τ .* p.qnetwork2[end].σ
    end





    # on policy actor update

    target_frac = p.target_frac #Float32(1.0/n_samples)
    τ_change = 3f-4

    μ_before, logσ_before = p.actor(p.device_rng, s)


    # acc_mags  = antithetic_mean_sac2(p, s, α; use_grad_a_q_norms = true)

    # w = q_rank_mask(acc_mags; target_frac=target_frac)

    # actor_inds = findall(x -> x<0.5, w)
    # fear_inds = findall(x -> x>=0.5, w)


    for i in 1:p.on_policy_actor_loops
        # Train Policy
        p_grad = Flux.gradient(p.actor) do actor
            aa, logp_π, μ, logσ = actor(p.device_rng, s; is_sampling=true, is_return_log_prob=true, is_return_params=true)
            q_input = vcat(s, aa)
            q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
            reward = mean(q)
            entropy = mean(logp_π)

            ignore_derivatives() do
                p.last_reward_term = reward
                p.last_entropy_term = α * entropy
                p.last_actor_loss = α * entropy - reward

            end


            # KL(new || old) für diagonale Gauß-Policy, pro Sample
            σ2   = exp.(2f0 .* logσ)
            σ0_2 = exp.(2f0 .* logσ_before)
            t1 = (σ2 ./ σ0_2)
            t2 = ((μ .- μ_before).^2) ./ σ0_2
            KLs = vec(sum(0.5f0 .* (t1 .+ t2 .- 1f0 .+ 2f0 .* (logσ_before .- logσ)); dims=1))
            KLs = Float32.(KLs)

            # weiche 70/30-Maske aus Q-Rängen (stopgrad)
            # w = ignore_derivatives() do
            #     # q_host = collect(Float32.(q))[:]
            #     # q_rank_mask(q_host; target_frac=target_frac)

            #     mags = grad_a_q_norms(p.qnetwork1, p.qnetwork2, s, aa)
            #     q_rank_mask(mags; target_frac=target_frac)
            # end

            #@show w

            # fear_term = p.fear_factor * mean(w .* KLs)


            # new experimental loss

            # actor_inds = findall(x -> x<0.5, w)
            # fear_inds = findall(x -> x>=0.5, w)

            # if (length(actor_inds) / length(w)) - target_frac > 0.05
            #     @show length(actor_inds) / length(w)
            # end

            #reward = mean(q[actor_inds])
            #entropy = mean(logp_π[actor_inds])

            # fear_term = p.fear_factor * mean(KLs[fear_inds])
            # fear_term = p.fear_factor * mean((μ .- μ_before).^2) # only mean value
            if p.target_frac != 1.0f0
                fear_term = p.fear_factor * mean(KLs)
            else
                fear_term = 0.0f0
            end

            #@show target_frac, length(actor_inds), length(fear_inds)

            actor_term = α * entropy - reward

            loss = actor_term + fear_term

            loss
        end
        Flux.update!(p.actor_state_tree, p.actor, p_grad[1])
    end


    μ_new, logσ_new = p.actor(p.device_rng, s)

    σ2   = exp.(2f0 .* logσ_new)
    σ0_2 = exp.(2f0 .* logσ_before)
    t1 = (σ2 ./ σ0_2)
    t2 = ((μ_new .- μ_before).^2) ./ σ0_2
    KLs = vec(sum(0.5f0 .* (t1 .+ t2 .- 1f0 .+ 2f0 .* (logσ_before .- logσ_new)); dims=1))
    KLs = Float32.(KLs)

    frac_changed = mean(KLs .> τ_change)
    ff_before = deepcopy(p.fear_factor)
    p.fear_factor = adjust_fear_factor(p.fear_factor, frac_changed; target=target_frac)

    if p.verbose
        #@show p.fear_factor - ff_before
        @show mean(logσ_before), mean(logσ_new), mean(μ_before), mean(μ_new)
        @show frac_changed
        @show p.fear_factor
    end

    # Tune entropy automatically
    if p.automatic_entropy_tuning
        # p.log_α -= p.lr_alpha * p.α * mean(-logp_π′ .- p.target_entropy)

        log_α_grad = Flux.gradient(p.log_α) do log_α
            exp(log_α[1]) .* mean(-logp_π′ .- p.target_entropy)
        end

        α_now = exp(p.log_α[1])
        g_scaled = log_α_grad ./ max(α_now, 1e-12)
        Flux.update!(p.log_α_state_tree, p.log_α, g_scaled[1])

        clamp!(p.log_α, -12.5f0, 1.5f0)

        p.α = exp.(p.log_α)[1]

    end


    if p.verbose
        println("Reward Term = ",  p.last_reward_term)
        println("Entropy Term = ",  p.last_entropy_term)

        println("mean(-logπ′) = ",  mean(-logp_π′))
        println("target_entropy = ", p.target_entropy)
        println("alpha = ",  p.α)
    end

end



function _update!(p::SACPolicy2, t::AbstractTrajectory)
    #this is for imitation learning

    if length(size(p.action_space)) == 2
        number_actuators = size(p.action_space)[2]
    else
        #mono case 
        number_actuators = 1
    end

    for i = 1:p.update_loops
        inds, batch = pde_sample(p.rng, t, BatchSampler{SARTS}(p.batch_size), number_actuators)
        update!(p, batch)
    end
end



function update!(p::SACPolicy2, batch::NamedTuple{SARTS})
    s, a, r, t, next_states = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α = p.γ, p.τ, p.α


    check_state_trees(p)


    acc_mu, acc_logp = antithetic_mean_sac2(p, next_states, α)

    next_values = acc_mu

    n_envs = size(t, 1)
    next_values = reshape( next_values, n_envs, :)


    y = r .+ γ .* (1 .- t) .* next_values

    
    logp_π′ = acc_logp

    p.last_target_q_mean = mean(y)
    p.last_mean_minus_log_pi = mean(-logp_π′)

    # Train Q Networks
    q_input = vcat(s, a)

    # println(repr(y))
    # error("abb")

    q_grad_1 = Flux.gradient(p.qnetwork1) do qnetwork1
        q1 = dropdims(qnetwork1(q_input), dims=1)

        ignore_derivatives() do
            p.last_q1_mean = mean(q1)
            p.last_critic1_loss = Flux.mse(q1, y)
        end

        Flux.mse(q1, y)
    end
    Flux.update!(p.qnetwork1_state_tree, p.qnetwork1, q_grad_1[1])
    if p.use_popart
        update!(p.qnetwork1.layers[end], y) 
    end

    q_grad_2 = Flux.gradient(p.qnetwork2) do qnetwork2
        q2 = dropdims(qnetwork2(q_input), dims=1)

        ignore_derivatives() do
            p.last_q2_mean = mean(q2)
            p.last_critic2_loss = Flux.mse(q2, y)
        end

        Flux.mse(q2, y)
    end
    Flux.update!(p.qnetwork2_state_tree, p.qnetwork2, q_grad_2[1])
    if p.use_popart
        update!(p.qnetwork2.layers[end], y) 
    end





    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end

    # Functors.fmap(p.target_qnetwork1, p.qnetwork1) do tgt, src
    #     if tgt isa AbstractArray && src isa AbstractArray
    #         @. tgt = (1 - τ) * tgt + τ * src
    #     end
    #     tgt
    # end

    # Functors.fmap(p.target_qnetwork2, p.qnetwork2) do tgt, src
    #     if tgt isa AbstractArray && src isa AbstractArray
    #         @. tgt = (1 - τ) * tgt + τ * src
    #     end
    #     tgt
    # end

    if p.use_popart
        # PopArt Polyak
        p.target_qnetwork1[end].μ = (1 - τ) .* p.target_qnetwork1[end].μ .+ τ .* p.qnetwork1[end].μ
        p.target_qnetwork1[end].σ = (1 - τ) .* p.target_qnetwork1[end].σ .+ τ .* p.qnetwork1[end].σ
        p.target_qnetwork2[end].μ = (1 - τ) .* p.target_qnetwork2[end].μ .+ τ .* p.qnetwork2[end].μ
        p.target_qnetwork2[end].σ = (1 - τ) .* p.target_qnetwork2[end].σ .+ τ .* p.qnetwork2[end].σ
    end
end