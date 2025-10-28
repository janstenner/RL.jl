
Base.@kwdef mutable struct SACPolicy2 <: AbstractPolicy
    actor::GaussianNetwork
    qnetwork1
    qnetwork2
    target_qnetwork1
    target_qnetwork2

    optimizer_actor = ADAM()
    optimizer_qnetwork1 = ADAM()
    optimizer_qnetwork2 = ADAM()
    actor_state_tree = nothing
    qnetwork1_state_tree = nothing
    qnetwork2_state_tree = nothing

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


function create_agent_sac2(;action_space, state_space, use_gpu = false, rng, y, t =0.005f0, a =0.2f0, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = gelu, fun_critic = nothing, tanh_end = false, n_agents = 1, logσ_is_network = false, batch_size = 32, start_steps = -1, start_policy = nothing, update_after = 1000, update_freq = 50, update_loops = 1, max_σ = 2.0f0, clip_grad = 0.5, start_logσ = 0.0, betas = (0.9, 0.999), trajectory_length = 10_000, automatic_entropy_tuning = true, lr_alpha = nothing, target_entropy = nothing, use_popart = false)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    isnothing(target_entropy)           &&  (target_entropy = -Float32(na))

    isnothing(lr_alpha)                 && (lr_alpha = Float32(learning_rate))


    qnetwork1 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic, is_critic2 = true, popart = use_popart)
    target_qnetwork1 = deepcopy(qnetwork1)

    qnetwork2 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic, is_critic2 = true, popart = use_popart)
    target_qnetwork2 = deepcopy(qnetwork2)

    Agent(
        policy = SACPolicy2(
           actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                ),
            qnetwork1 = qnetwork1,
            qnetwork2 = qnetwork2,
            target_qnetwork1 = target_qnetwork1,
            target_qnetwork2 = target_qnetwork2,

            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
            optimizer_qnetwork1 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
            optimizer_qnetwork2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),

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
            use_popart = use_popart
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
    p.update_step % p.update_freq == 0 || return

    #println("UPDATE!")
    for i = 1:p.update_loops
        inds, batch = pde_sample(p.rng, t, BatchSampler{SARTS}(p.batch_size), number_actuators)
        update!(p, batch)
    end
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
function adjust_fear_factor(fear_factor, frac_changed; target=0.30f0, up=1.20f0, down=0.85f0)
    frac_changed > target ? fear_factor*up : fear_factor*down
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
    s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α = p.γ, p.τ, p.α

    a′, logp_π′ = p.actor(p.device_rng, s′; is_sampling=true, is_return_log_prob=true)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* dropdims(q′ .- α .* logp_π′, dims=1)

    p.last_target_q_mean = mean(y)
    p.last_mean_minus_log_pi = mean(-logp_π′)


    if isnothing(p.actor_state_tree) || isnothing(p.qnetwork1_state_tree) || isnothing(p.qnetwork2_state_tree)
        println("________________________________________________________________________")
        println("Reset Optimizers")
        println("________________________________________________________________________")
        p.actor_state_tree = Flux.setup(p.optimizer_actor, p.actor)
        p.qnetwork1_state_tree = Flux.setup(p.optimizer_qnetwork1, p.qnetwork1)
        p.qnetwork2_state_tree = Flux.setup(p.optimizer_qnetwork2, p.qnetwork2)

        p.log_α_state_tree = Flux.setup(p.optimizer_log_α, p.log_α)
    end

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


    if p.update_step % (p.update_freq * 10) == 0

        target_frac = 0.30f0
        τ_change = 1f-5

        μ_before, logσ_before = p.actor(p.device_rng, s)

        # Train Policy
        p_grad = Flux.gradient(p.actor) do actor
            a, logp_π, μ, logσ = actor(p.device_rng, s; is_sampling=true, is_return_log_prob=true, is_return_params=true)
            q_input = vcat(s, a)
            q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
            reward = mean(q)
            entropy = mean(logp_π)

            ignore_derivatives() do
                p.last_reward_term = reward
                p.last_entropy_term = α * entropy
                p.last_actor_loss = α * entropy - reward

            end

            #fear = mean( (exp.(logp_π - logp_π_fixed) .- 1).^2 ) .* p.fear_factor

            actor_term = α * entropy - reward

            μ_old    = Zygote.dropgrad(μ)
            logσ_old = Zygote.dropgrad(logσ)

            # KL(new || old) für diagonale Gauß-Policy, pro Sample
            σ2   = exp.(2f0 .* logσ)
            σ0_2 = exp.(2f0 .* logσ_old)
            t1 = (σ2 ./ σ0_2)
            t2 = ((μ .- μ_old).^2) ./ σ0_2
            KLs = vec(sum(0.5f0 .* (t1 .+ t2 .- 1f0 .+ 2f0 .* (logσ_old .- logσ)); dims=1))
            KLs = Float32.(KLs)

            # weiche 70/30-Maske aus Q-Rängen (stopgrad)
            w = ignore_derivatives() do
                q_host = collect(Float32.(q))[:]
                q_rank_mask(q_host; target_frac=target_frac)
            end

            fear_term = p.fear_factor * mean(w .* KLs)

            loss = actor_term + fear_term

            loss
        end
        Flux.update!(p.actor_state_tree, p.actor, p_grad[1])


        μ_new, logσ_new = p.actor(p.device_rng, s)

        σ2   = exp.(2f0 .* logσ_new)
        σ0_2 = exp.(2f0 .* logσ_before)
        t1 = (σ2 ./ σ0_2)
        t2 = ((μ_new .- μ_before).^2) ./ σ0_2
        KLs = vec(sum(0.5f0 .* (t1 .+ t2 .- 1f0 .+ 2f0 .* (logσ_before .- logσ_new)); dims=1))
        KLs = Float32.(KLs)

        frac_changed = mean(KLs .> τ_change)
        #@show frac_changed
        ff_before = deepcopy(p.fear_factor)
        p.fear_factor = adjust_fear_factor(p.fear_factor, frac_changed; target=target_frac, up=1.01f0, down=0.99f0)

        #@show p.fear_factor - ff_before
        @show p.fear_factor
    end

    


    # Tune entropy automatically
    if p.automatic_entropy_tuning
        # p.log_α -= p.lr_alpha * p.α * mean(-logp_π′ .- p.target_entropy)

        log_α_grad = Flux.gradient(p.log_α) do log_α
            exp(log_α[1]) .* mean(-logp_π′ .- p.target_entropy)
        end
        Flux.update!(p.log_α_state_tree, p.log_α, log_α_grad[1])

        clamp!(p.log_α, -6.5, Inf)

        p.α = exp.(p.log_α)[1]

    end


    if p.update_step % (p.update_freq * 100) == 0
        println("Reward Term = ",  p.last_reward_term)
        println("Entropy Term = ",  p.last_entropy_term)

        println("mean(-logπ′) = ",  mean(-logp_π′))
        println("target_entropy = ", p.target_entropy)
        println("alpha = ",  p.α)
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
end