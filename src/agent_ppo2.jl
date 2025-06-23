Base.@kwdef mutable struct ModulationModule
    # Configuration parameters
    p_explore::Float32 = 1/140   # probability to enter explore mode each time step
    min_amp::Float32    = 0.1    # minimum exploration amplitude
    max_amp::Float32    = 1.2    # maximum exploration amplitude
    min_width::Int      = 5      # minimum explore duration (in steps)
    max_width::Int      = 80     # maximum explore duration (in steps)

    # Internal state
    state::Symbol       = :idle  # current mode (:idle or :explore)
    last::Float32       = 0.0    # last output value

    # Explore-mode parameters (set upon entering explore)
    amp::Float32        = 1.0    # current amplitude
    width::Int          = 0      # total duration of current explore burst
    sigma::Float32      = 1.0    # Gaussian standard deviation
    center::Float32     = 0.0    # Gaussian center
    t::Int              = 0      # time step within current explore burst
    min_value::Float32 = 0.0    # minimum value for the output
end


function (mm::ModulationModule)()
    # mm.last = 1.0f0
    # return mm.last

    if mm.state == :idle
        # In idle mode, output zero
        mm.last = mm.min_value
        # Possibly switch to explore
        if rand() < mm.p_explore
            mm.state  = :explore
            mm.width  = rand(mm.min_width:mm.max_width)
            mm.amp    = mm.min_amp + rand() * (mm.max_amp - mm.min_amp)
            mm.sigma  = mm.width / 6
            mm.center = mm.width / 2
            mm.t      = 0
        end

    elseif mm.state == :explore
        # Compute Gaussian-shaped output
        mm.last = mm.min_value + mm.amp * exp(-((mm.t - mm.center)^2) / (2 * mm.sigma^2))
        mm.t += 1
        # Return to idle after the burst completes
        if mm.t > mm.width
            mm.state = :idle
        end
    end

    return mm.last
end


Base.@kwdef mutable struct ActorCritic2{A,C}
    actor::A
    critic::C
    optimizer_actor = ADAM()
    optimizer_sigma = ADAM()
    optimizer_critic = ADAM()
    actor_state_tree = nothing
    sigma_state_tree = nothing
    critic_state_tree = nothing
end

@forward ActorCritic2.critic device

function create_agent_ppo2(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = relu, fun_critic = nothing, tanh_end = false, n_envs = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = false, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, critic_regularization_loss_weight=0.01f0, logσ_regularization_loss_weight=0.01f0, adaptive_weights = false, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, noise = nothing, noise_scale = 90, fear_factor = 1.0)

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

    Agent(
        policy = PPOPolicy2(
            approximator = isnothing(approximator) ? ActorCritic2(
                actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                ),
                critic = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, fun = fun_critic),
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
                optimizer_sigma = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
            ) : approximator,
            γ = y,
            λ = p,
            clip_range = clip_range,
            max_grad_norm = 0.5f0,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            critic_regularization_loss_weight = critic_regularization_loss_weight,
            logσ_regularization_loss_weight = logσ_regularization_loss_weight,
            adaptive_weights = adaptive_weights,
            dist = Normal,
            rng = rng,
            update_freq = update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            noise = noise,
            noise_sampler = noise_sampler,
            noise_scale = noise_scale,
            fear_factor = fear_factor,
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
                next_values = Float32 => (1, n_envs),
        ),
    )
end



"""
    PPOPolicy2(;kwargs)

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


mutable struct PPOPolicy2{A<:ActorCritic2,D,R} <: AbstractPolicy
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
    critic_regularization_loss_weight::Float32
    logσ_regularization_loss_weight::Float32
    adaptive_weights::Bool
    rng::R
    n_random_start::Int
    update_freq::Int
    update_step::Int
    clip1::Bool
    normalize_advantage::Bool
    start_steps
    start_policy
    target_kl
    noise
    noise_sampler
    noise_scale
    noise_step
    fear_factor
    mm
    critic_target
    last_action_log_prob::Vector{Float32}
    last_sigma::Vector{Float32}
    last_mu::Vector{Float32}
end

function PPOPolicy2(;
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
    critic_regularization_loss_weight=0.01f0,
    logσ_regularization_loss_weight=0.01f0,
    adaptive_weights = true,
    dist=Normal,
    rng=Random.GLOBAL_RNG,
    clip1 = false,
    normalize_advantage = false,
    start_steps = -1,
    start_policy = nothing,
    target_kl = 100.0,
    noise = nothing,
    noise_sampler = nothing,
    noise_scale = 90.0,
    noise_step = 0,
    fear_factor = 1.0f0,
    mm = ModulationModule(),
    critic_target = 0.0f0,
)
    PPOPolicy2{typeof(approximator),dist,typeof(rng)}(
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
        critic_regularization_loss_weight,
        logσ_regularization_loss_weight,
        adaptive_weights,
        rng,
        n_random_start,
        update_freq,
        update_step,
        clip1,
        normalize_advantage,
        start_steps,
        start_policy,
        target_kl,
        noise,
        noise_sampler,
        noise_scale,
        noise_step,
        fear_factor,
        mm,
        critic_target,
        [0.0],
        [0.0],
        [0.0],
    )
end

function prob(
    p::PPOPolicy2{<:ActorCritic2{<:GaussianNetwork},Normal},
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

function prob(p::PPOPolicy2{<:ActorCritic2,Categorical}, state::AbstractArray, mask)
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

function prob(p::PPOPolicy2, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::PPOPolicy2, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::PPOPolicy2)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end



function (p::PPOPolicy2)(env::AbstractEnv)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        dist = prob(p, env)

        modulation_value = p.mm()
        dist.σ .*= modulation_value
        dist.σ .+= modulation_value * 0.25

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
    policy::PPOPolicy2,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(
        trajectory;
        state = state(env),
        action = action,
        action_log_prob = policy.last_action_log_prob,
        explore_mod = policy.mm.last,
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy2,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)[:]

    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
    push!(trajectory[:next_values], policy.approximator.critic(send_to_device(device(policy.approximator), env.state)) |> send_to_host)
end

function update!(
    p::PPOPolicy2,
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









"""
    sample_negatives_far(x_batch; m=size(x_batch,2), δ=0.1,
                         max_attempts=1_000, sample_fn=()->randn(eltype(x_batch), size(x_batch,1)))

Draw up to `m` vectors in ℝⁿ (same n as `x_batch`) so that each new vector is at least
Euclidean distance `δ` from *every* column of `x_batch`.  By default samples
from N(0,1). Throws an error if it fails `max_attempts` times for any sample.
"""
function sample_negatives_far(x_batch::AbstractMatrix{T};
                              m::Int=size(x_batch,2),
                              δ::Real=0.1,
                              max_attempts::Int=1_000,
                              sample_fn::Function=()->randn(eltype(x_batch), size(x_batch,1))) where {T}

    δ2 = δ^2
    x_neg = Vector{Vector{T}}() 

    for i in 1:m
        attempt = 0
        while max_attempts > attempt
            attempt += 1
            cand = sample_fn()
            # compute squared distances to every column in x_batch
            d2 = sum((x_batch .- cand).^2; dims=1)
            if minimum(d2) ≥ δ2
                push!(x_neg, cand)
                break
            end
        end
    end

    return hcat(x_neg...)
end



function _update!(p::PPOPolicy2, t::Any)
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
    w₄ = p.critic_regularization_loss_weight
    w₅ = p.logσ_regularization_loss_weight
    D = device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_rollout = size(t[:terminal])
    @assert n_envs * n_rollout % n_microbatches == 0 "size mismatch"
    microbatch_size = n_envs * n_rollout ÷ n_microbatches

    n = length(t)
    states = to_device(t[:state])


    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))

    values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)
    next_values = reshape(flatten_batch(t[:next_values]), n_envs, :)

    advantages, returns = generalized_advantage_estimation(
        t[:reward],
        values,
        next_values,
        γ,
        λ;
        dims=2,
        terminal=t[:terminal]
    )
    # returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    advantages = to_device(advantages)

    # if p.normalize_advantage
    #     advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
    # end

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)
    explore_mod = to_device(t[:explore_mod])

    stop_update = false

    actor_losses = Float32[]
    critic_losses = Float32[]
    entropy_losses = Float32[]
    logσ_regularization_losses = Float32[]
    critic_regularization_losses = Float32[]

    excitements = Float32[]
    fears = Float32[]

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))
            exp_m = to_device(collect(select_last_dim(explore_mod, inds)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds]
            adv = vec(advantages)[inds]

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end



            if isnothing(AC.actor_state_tree)
                AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor.μ)
            end

            if isnothing(AC.sigma_state_tree)
                AC.sigma_state_tree = Flux.setup(AC.optimizer_sigma, AC.actor.logσ)
            end

            if isnothing(AC.critic_state_tree)
                AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
            end

            # s_neg = sample_negatives_far(s)

            g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic
                v′ = critic(s) |> vec
                
                μ, logσ = actor(s)

                σ = (exp.(logσ) .* exp_m) .+ (exp_m .* 0.25)

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

                # if !(isempty(s_neg))
                #     v_neg = critic(s_neg)
                #     critic_regularization = mean((v_neg .- p.critic_target) .^ 2)
                # else
                #     critic_regularization = 0.0f0
                # end

                # if AC.actor.logσ_is_network
                #     logσ_neg = AC.actor.logσ(s_neg)
                #     logσ_regularization  = mean((logσ_neg .- log(AC.actor.max_σ)) .^ 2)	
                # else
                #     logσ_regularization  = 0.0f0
                # end

                actor_loss = -mean(((ratio .* adv) - fear)) # .* exp_m[:])

                # surr1 = ratio .* adv
                # surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                # actor_loss = -mean(min.(surr1, surr2) .* exp_m[:])

                critic_loss = mean(((r .- v′) .^ 2)) # .* exp_m[:])
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss #+ w₄ * critic_regularization #+ w₅ * logσ_regularization


                ignore() do
                    # println("---------------------")
                    # println(critic_loss)
                    # println(critic_regularization)
                    # println("---------------------")
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)
                    # push!(critic_regularization_losses, w₄ * critic_regularization)
                    # push!(logσ_regularization_losses, w₅ * logσ_regularization)

                    push!(excitements, maximum(adv))
                    push!(fears, maximum(fear))

                    # polyak update critic target
                    p.critic_target = p.critic_target * 0.9 + (maximum(r) - 0.1) * 0.1
                end

                loss
            end
            
            if !stop_update
                Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
                Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
                Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
            else
                break
            end

        end

        if stop_update
            break
        end
    end


    # everything here is just magnitude (abs), not real mean

    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic_loss = mean(abs.(critic_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    # mean_logσ_regularization_loss = mean(abs.(logσ_regularization_losses))
    # mean_critic_regularization_loss = mean(abs.(critic_regularization_losses))
    
    println("---")
    println("mean actor loss: $(mean_actor_loss)")
    println("mean critic loss: $(mean_critic_loss)")
    println("mean entropy loss: $(mean_entropy_loss)")
    # println("mean logσ regularization loss: $(mean_logσ_regularization_loss)")
    # println("mean critic regularization loss: $(mean_critic_regularization_loss)")

    max_excitement = maximum(abs.(excitements))
    max_fear = maximum(abs.(fears))
    
    println("max excitement: $(max_excitement)")
    println("max fear: $(max_fear)")

    if p.adaptive_weights
        # actor_factor = clamp(1.0/mean_actor_loss, 0.99, 1.01)
        # critic_factor = clamp(0.5/mean_critic_loss, 0.99, 1.01)
        # entropy_factor = clamp(0.01/mean_entropy_loss, 0.99, 1.01)
        # logσ_regularization_factor = clamp(0.1/mean_logσ_regularization_loss, 0.9, 1.1)
        # critic_regularization_factor = clamp(0.3*mean_critic_loss/mean_critic_regularization_loss, 0.9, 1.1)

        # fear_factor_factor = clamp(((max_excitement * 0.04005) / (max_fear)), 0.5, 1.01)

        new_factor_factor = max_excitement * 0.01


        # println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
        # println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
        # println("changing entropy weight from $(w₃) to $(w₃*entropy_factor)")
        # println("changing logσ regularization weight from $(w₅) to $(w₅*logσ_regularization_factor)")
        # println("changing critic regularization weight from $(w₄) to $(w₄*critic_regularization_factor)")
        println("changing fear factor from $(p.fear_factor) to $(p.fear_factor * 0.9 + new_factor_factor * 0.1)")

        # println("current critic_target is $(p.critic_target)")

        # p.actor_loss_weight = w₁ * actor_factor
        # p.critic_loss_weight = w₂ * critic_factor
        # p.entropy_loss_weight = w₃ * entropy_factor
        # p.logσ_regularization_loss_weight = w₅ * logσ_regularization_factor
        # p.critic_regularization_loss_weight = w₄ * critic_regularization_factor

        # polyak update fear_factor
        p.fear_factor = p.fear_factor * 0.9 + new_factor_factor * 0.1
    end

    println("---")
end

