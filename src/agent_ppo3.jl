
Base.@kwdef mutable struct PPOPolicy3 <: AbstractPolicy
    approximator
    trajectory_no_exploration
    exploration_mode::Bool = false
    γ::Float32 = 0.99f0
    λ::Float32 = 0.95f0
    polyak_factor::Float32 = 0.995f0
    clip_range::Float32 = 0.2f0
    n_microbatches::Int = 4
    n_epochs::Int = 4
    actor_loss_weight::Float32 = 1.0f0
    critic_loss_weight::Float32 = 0.5f0
    entropy_loss_weight::Float32 = 0.01f0
    critic_regularization_loss_weight::Float32 = 0.01f0
    logσ_regularization_loss_weight::Float32 = 0.01f0
    adaptive_weights::Bool = true
    rng = Random.GLOBAL_RNG
    update_freq::Int = 256
    update_freq_no_exploration::Int = 256
    update_step::Int = 0
    update_step_no_exploration::Int = 0
    n_updates::Int = 0
    clip1::Bool = false
    normalize_advantage::Bool = false
    start_steps = -1
    start_policy = nothing
    target_kl = 100.0
    noise = nothing
    noise_sampler = nothing
    noise_scale = 90.0
    noise_step = 0
    fear_factor = 1.0f0
    fear_scale = 0.5f0
    new_loss = true
    mm = ModulationModule()
    critic_target = 0.0f0
    last_action_log_prob::Vector{Float32} = [0.0f0]
    last_sigma::Vector{Float32} = [0.0f0]
    last_mu::Vector{Float32} = [0.0f0]
end


Base.@kwdef mutable struct ActorCritic3{A,C,C2,C3}
    actor::A
    critic::C # value function approximator
    critic2::C2 # action value function approximator minus r
    critic3::C3 # value function approximator for exploration
    critic_target::C
    critic2_target::C2
    critic3_target::C3
    optimizer_actor = ADAM()
    optimizer_sigma = ADAM()
    optimizer_critic = ADAM()
    optimizer_critic2 = ADAM()
    optimizer_critic3 = ADAM()
    actor_state_tree = nothing
    sigma_state_tree = nothing
    critic_state_tree = nothing
    critic2_state_tree = nothing
    critic3_state_tree = nothing
end

@forward ActorCritic3.critic device



function create_agent_ppo3(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = relu, fun_critic = nothing, tanh_end = false, n_envs = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = false, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, critic_regularization_loss_weight=0.01f0, logσ_regularization_loss_weight=0.01f0, adaptive_weights = false, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, noise = nothing, noise_scale = 90, fear_factor = 1.0, fear_scale = 0.5, new_loss = true, update_freq_no_exploration = 256)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    if noise == "perlin"
        noise_sampler = perlin_2d(; seed=Int(floor(rand(rng) * 1e12)))
    else
        noise_sampler = nothing
    end

    mm = ModulationModule()

    critic = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic)
    critic2 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic, is_critic2 = true)
    critic3 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic)

    Agent(
        policy = PPOPolicy3(
            approximator = isnothing(approximator) ? ActorCritic3(
                actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                ),
                critic = critic,
                critic2 = critic2,
                critic3 = critic3,
                critic_target = deepcopy(critic),
                critic2_target = deepcopy(critic2),
                critic3_target = deepcopy(critic3),
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate/8, betas)),
                optimizer_sigma = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
                optimizer_critic2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
                optimizer_critic3 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
            ) : approximator,
            trajectory_no_exploration = CircularArrayTrajectory(;
                capacity = update_freq_no_exploration,
                state = Float32 => (size(state_space)[1], n_envs),
                reward = Float32 => (n_envs,),
                terminal = Bool => (n_envs,),
                next_states = Float32 => (size(state_space)[1], n_envs),
            ),
            γ = y,
            λ = p,
            clip_range = clip_range,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            critic_regularization_loss_weight = critic_regularization_loss_weight,
            logσ_regularization_loss_weight = logσ_regularization_loss_weight,
            adaptive_weights = adaptive_weights,
            rng = rng,
            update_freq = update_freq,
            update_freq_no_exploration = update_freq_no_exploration,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            noise = noise,
            noise_sampler = noise_sampler,
            noise_scale = noise_scale,
            fear_factor = fear_factor,
            fear_scale = fear_scale,
            new_loss = new_loss,
            mm = mm,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_envs),
                action = Float32 => (size(action_space)[1], n_envs),
                action_log_prob = Float32 => (n_envs),
                reward = Float32 => (n_envs),
                explore_mod = Float32 => (n_envs),
                terminal = Bool => (n_envs,),
                next_states = Float32 => (size(state_space)[1], n_envs),
        ),
    )
end





function prob(p::PPOPolicy3, state::AbstractArray)
    μ, logσ = p.approximator.actor(send_to_device(device(p.approximator), state)) |> send_to_host

    StructArray{Normal}((μ, exp.(logσ)))
end

function prob(p::PPOPolicy3, env::AbstractEnv)
    prob(p, state(env))
end


function (p::PPOPolicy3)(env::AbstractEnv; ignore_explore_mode = false)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        dist = prob(p, env)

        modulation_value = p.mm()
        dist.σ .*= modulation_value
        #dist.σ .+= modulation_value * 0.25

        if p.clip1
            clamp!(dist.μ, -1.0, 1.0)
        end

        if p.exploration_mode || ignore_explore_mode
            if isnothing(p.noise)
                action = rand.(p.rng, dist)
            else
                norm_factor = float(pi / 20) #* 5
                p.noise_step += 1
                noise = [CoherentNoise.sample(p.noise_sampler, p.noise_step/p.noise_scale, float(π+2*i))/norm_factor for i in 1:size(dist.μ, 2)]
                action = dist.μ + dist.σ .* noise'
            end

            if p.clip1
                clamp!(action, -1.0, 1.0)
            end
        else
            action = dist.μ
        end

        # put the last action log prob behind the clip
        
        ###p.last_action_log_prob = vec(sum(logpdf.(dist, action), dims=1))

        if ndims(action) == 2
            log_p = vec(sum(normlogpdf(dist.μ, dist.σ, action), dims=1))
        else
            log_p = normlogpdf(dist.μ, dist.σ, action)
        end

        p.last_action_log_prob = log_p
        p.last_mu = dist.μ[:]
        p.last_sigma = dist.σ[:]

        action
    end
end








function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy3,
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
            explore_mod = policy.mm.last,
        )
    else
        push!(
            policy.trajectory_no_exploration;
            state = state(env),
        )
    end
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy3,
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

        push!(policy.trajectory_no_exploration[:reward], r)
        push!(policy.trajectory_no_exploration[:terminal], is_terminated(env))
        push!(policy.trajectory_no_exploration[:next_states], state(env))
    end
end

function update!(
    p::PPOPolicy3,
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
            p.update_step_no_exploration = 0
        end
    else
        p.update_step_no_exploration += 1

        if p.update_step_no_exploration % p.update_freq_no_exploration == 0
            _update_no_exploration!(p)
            p.exploration_mode = true
            p.update_step = 0
        end
    end
    
end



function calculate_returns(rewards, terminal, next_values, γ)
    n_envs, n_rollout = size(terminal)
    returns = zeros(Float32, n_envs, n_rollout)

    for i in length(rewards):-1:1
        is_continue = .!terminal[:,i]

        if i == length(rewards)
            returns[:,i] = rewards[:,i] + γ .* next_values[:,i] .* is_continue
        else
            returns[:,i] = rewards[:,i] + γ .* returns[:,i + 1] .* is_continue
        end
    end

    returns
end



function _update_no_exploration!(p::PPOPolicy3)

    println("TRAIN_NO_EXPLORATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    t = p.trajectory_no_exploration
    rng = p.rng
    AC = p.approximator
    γ = p.γ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    w₂ = p.critic_loss_weight
    D = device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_rollout = size(t[:terminal])

    microbatch_size = Int(floor(n_envs * n_rollout ÷ n_microbatches))

    n = length(t)
    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))
    next_states = to_device(t[:next_states])

    # calculate returns
    rewards = to_device(t[:reward])
    terminal = to_device(t[:terminal])
    next_values = reshape(send_to_host(AC.critic_target(flatten_batch(next_states))), n_envs, :)
    returns = calculate_returns(rewards, terminal, next_values, γ)


    critic_losses = Float32[]

    if p.n_updates == 1 || isnothing(AC.critic_state_tree)
        println("__________________________________________________________________________________________________________________________")
        println("Reset Optimizers")
        println("__________________________________________________________________________________________________________________________")
        AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
    end

    # AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            ret = vec(returns)[inds]
            rew = vec(rewards)[inds]
            nv = vec(next_values)[inds]

            g_critic = Flux.gradient(AC.critic) do critic
                v′ = critic(s) |> vec

                critic_loss = mean(((rew + nv) .- v′) .^ 2)

                loss = w₂ * critic_loss

                ignore() do
                    push!(critic_losses, w₂ * critic_loss)
                end

                loss
            end
            
            Flux.update!(AC.critic_state_tree, AC.critic, g_critic[1])

            # update PopArt parameters
            update!(AC.critic.layers[end], rew + nv)

            # polyak averaging
            pf = p.polyak_factor
            for (dest, src) in zip(Flux.params([AC.critic_target]), Flux.params([AC.critic]))
                dest .= pf .* dest .+ (1 - pf) .* src
            end

            AC.critic.layers[end].μ = AC.critic_target.layers[end].μ * pf + (1 - pf) * AC.critic.layers[end].μ
            AC.critic.layers[end].σ = AC.critic_target.layers[end].σ * pf + (1 - pf) * AC.critic.layers[end].σ
        end
    end

    mean_critic_loss = mean(abs.(critic_losses))
    
    println("---")
    println("mean critic loss: $(mean_critic_loss)")
    println("---")
end


function _update!(p::PPOPolicy3, t::Any; IL=false)

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

    n_envs, n_rollout = size(t[:terminal])
    
    microbatch_size = Int(floor(n_envs * n_rollout ÷ n_microbatches))

    n = length(t)
    states = to_device(t[:state])
    next_states = to_device(t[:next_states])
    actions = to_device(t[:action])

    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))
    next_states_flatten_on_host = flatten_batch(select_last_dim(t[:next_states], 1:n))

    # values = reshape(send_to_host(AC.critic3(flatten_batch(states))), n_envs, :)

    mus = AC.actor.μ(states_flatten_on_host)
    offsets = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), mus) )) , n_envs, :)

    # calculate returns
    rewards = to_device(t[:reward])
    terminal = to_device(t[:terminal])
    next_values = reshape(send_to_host(AC.critic3(flatten_batch(next_states))), n_envs, :)
    returns = calculate_returns(rewards, terminal, next_values, γ)
    
    advantages = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), flatten_batch(actions)) )) , n_envs, :) - offsets
    advantages = to_device(advantages)

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)
    explore_mod = to_device(t[:explore_mod])

    stop_update = false

    actor_losses = Float32[]
    critic2_losses = Float32[]
    critic3_losses = Float32[]
    entropy_losses = Float32[]

    excitements = Float32[]
    fears = Float32[]



    if p.n_updates == 1  || isnothing(AC.actor_state_tree) || isnothing(AC.sigma_state_tree) || isnothing(AC.critic2_state_tree)
        println("__________________________________________________________________________________________________________________________")
        println("Reset Optimizers")
        println("__________________________________________________________________________________________________________________________")
        AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor.μ)
        AC.sigma_state_tree = Flux.setup(AC.optimizer_sigma, AC.actor.logσ)
        AC.critic2_state_tree = Flux.setup(AC.optimizer_critic2, AC.critic2)
        AC.critic3_state_tree = Flux.setup(AC.optimizer_critic3, AC.critic3)
    end

    # AC.critic2_state_tree = Flux.setup(AC.optimizer_critic2, AC.critic2)

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            ns = to_device(collect(select_last_dim(next_states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))
            exp_m = to_device(collect(select_last_dim(explore_mod, inds)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            log_p = vec(action_log_probs)[inds]
            adv = vec(advantages)[inds]
            ret = vec(returns)[inds]
            rew = vec(rewards)[inds]
            nv = vec(next_values)[inds]

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            # s_neg = sample_negatives_far(s)

            g_actor, g_critic3 = Flux.gradient(AC.actor, AC.critic3) do actor, critic3

                v′ = critic3(s) |> vec

                μ, logσ = actor(s)

                σ = (exp.(logσ) .* exp_m) #.+ (exp_m .* 0.25)

                if ndims(a) == 2
                    log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims=1))
                else
                    log_p′ₐ = normlogpdf(μ, σ, a)
                end

                #clamp!(log_p′ₐ, log(1e-8), Inf)

                entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                
                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl
                        println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        stop_update = true
                    end
                end

                fear = (abs.((ratio .- 1)) + ones(size(ratio))).^1.3 .* p.fear_factor

                if p.new_loss
                    actor_loss = -mean(((ratio .* adv) - fear)) # .* exp_m[:])
                else
                    surr1 = ratio .* adv
                    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))
                end

                if IL
                    critic3_loss = 0.0
                else
                    critic3_loss = mean(((rew + nv) .- v′) .^ 2)
                end


                loss = w₁ * actor_loss + w₂ * critic3_loss - w₃ * entropy_loss


                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)
                    push!(critic3_losses, w₂ * critic3_loss)

                    push!(excitements, maximum(adv))
                    push!(fears, maximum(fear))
                end

                loss
            end
            
            if !stop_update
                Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
                # Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
                Flux.update!(AC.critic3_state_tree, AC.critic3, g_critic3)


                # update PopArt parameters
                update!(AC.critic3.layers[end], rew + nv)

                # polyak averaging
                pf = p.polyak_factor
                for (dest, src) in zip(Flux.params([AC.critic3_target]), Flux.params([AC.critic3]))
                    dest .= pf .* dest .+ (1 - pf) .* src
                end

                AC.critic3.layers[end].μ = AC.critic3_target.layers[end].μ * pf + (1 - pf) * AC.critic3.layers[end].μ
                AC.critic3.layers[end].σ = AC.critic3_target.layers[end].σ * pf + (1 - pf) * AC.critic3.layers[end].σ
            else
                break
            end

            v′ = AC.critic(s) |> vec
            nv′ = AC.critic3(ns) |> vec

            g_critic2 = Flux.gradient(AC.critic2) do critic2
                critic2_values = critic2(vcat(s,a)) |> vec

                if IL
                    critic2_loss = 0.0
                else
                    critic2_loss = mean(((critic2_values .- (rew + γ * nv′ - v′)) .^ 2))
                end

                ignore() do
                    push!(critic2_losses, w₂ * critic2_loss)
                end

                w₂ * critic2_loss
            end

            Flux.update!(AC.critic2_state_tree, AC.critic2, g_critic2[1])

            update!(AC.critic2.layers[end], rew + γ * nv′ - v′) 

        end

        if stop_update
            break
        end
    end




    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic2_loss = mean(abs.(critic2_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    
    println("---")
    println("mean actor loss: $(mean_actor_loss)")
    println("mean critic2 loss: $(mean_critic2_loss)")
    println("mean entropy loss: $(mean_entropy_loss)")

    mean_excitement = mean(abs.(excitements))
    max_fear = maximum(abs.(fears))
    
    println("mean excitement: $(mean_excitement)")
    println("max fear: $(max_fear)")

    if p.new_loss && p.adaptive_weights
        new_factor_factor = mean_excitement * p.fear_scale

        println("changing fear factor from $(p.fear_factor) to $(p.fear_factor * 0.9 + new_factor_factor * 0.1)")

        p.fear_factor = p.fear_factor * 0.9 + new_factor_factor * 0.1
    end

    println("---")
end