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

function create_agent_ppo(;action_space, state_space, use_gpu, rng, y, p, update_freq, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, trajectory_length = 1000, learning_rate = 0.001, fun = relu, fun_critic = nothing, n_envs = 2)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)

    init = Flux.glorot_uniform(rng)

    reward_size = 1

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
            n_epochs = 10,
            n_microbatches = 32,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.00f0,
            dist = Normal,
            rng = rng,
            update_freq = update_freq,
        ),
        trajectory = PPOTrajectory(;
            capacity = update_freq,
            state = Float32 => (size(state_space)[1], n_envs),
            action = Float32 => (size(action_space)[1], n_envs),
            action_log_prob = Float32 => (size(action_space)[1], n_envs),
            reward = Float32 => (n_envs,),
            terminal = Bool => (n_envs,),
        ),
    )
end