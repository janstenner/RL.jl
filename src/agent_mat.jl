# ----------- Custom CrossAttention for MAT -----------


struct CustomCrossAttention{A, Q, KV, O} <: Transformers.Layers.LayerStruct
    attention_op::A
    q_proj::Q
    kv_proj::KV #::NSplit{StaticInt{2}, KV}
    o_proj::O
end
Flux.@functor CustomCrossAttention

function argument_names(ca::CustomCrossAttention)
    required_names = (:hidden_state, :memory)
    field_names = invoke(Transformers.Layers.argument_names, Tuple{Transformers.Layers.LayerStruct}, ca)
    cross_field_names = Transformers.Layers.remove_name(Transformers.Layers.prefix_name(:cross, field_names), :cross_hidden_state)
    return Base.merge_names(required_names, cross_field_names)
end

Transformers.Layers.argument_names(ca::CustomCrossAttention) = argument_names(ca)

function (ca::CustomCrossAttention)(nt::NamedTuple)
    hidden_state, memory = nt.hidden_state, nt.memory
    cross_attention_mask = ChainRulesCore.ignore_derivatives(()->get(nt, :cross_attention_mask, nothing))
    nt_ext = Base.structdiff(nt, NamedTuple{(:hidden_state, :memory, :attention_mask, :cross_attention_mask)})
    q = Transformers.Layers.with_extra(ca.q_proj, memory, nt_ext)
    kv = Transformers.Layers.with_extra(ca.kv_proj, hidden_state, nt_ext)
    _a = Transformers.Layers._apply_cross_attention_op(ca.attention_op, q, kv, cross_attention_mask)
    a = Transformers.Layers.rename(Base.structdiff(_a, NamedTuple{(:attention_mask, :cross_attention_mask)}),
               Val(:attention_score), Val(:cross_attention_score))
    y = Transformers.Layers.apply_on_namedtuple(ca.o_proj, a)
    return merge(nt, y)
end


function CustomCrossAttention(head::Int, hidden_size::Int; dropout = nothing, return_score = false)
    @assert rem(hidden_size, head) == 0 "`hidden_size` should be dividible by `head` if `head_hidden_size` is not set"
    head_hidden_size = div(hidden_size, head)
    return CustomCrossAttention(head, hidden_size, head_hidden_size; dropout, return_score)
end
function CustomCrossAttention(head::Int, hidden_size::Int, head_hidden_size::Int; dropout = nothing, return_score = false)
    cross_atten_op = Transformers.Layers.MultiheadQKVAttenOp(head, dropout)
    return_score && (cross_atten_op = NeuralAttentionlib.WithScore(cross_atten_op))
    ca = CustomCrossAttention(cross_atten_op, head, hidden_size, head_hidden_size)
    return ca
end

function CustomCrossAttention(cross_atten_op, head::Int, hidden_size::Int, head_hidden_size::Int)
    c_q_proj = Dense(hidden_size, head * head_hidden_size)
    c_kv_proj = Dense(hidden_size, 2head * head_hidden_size)
    c_o_proj = Dense(head * head_hidden_size, hidden_size)
    ca = CustomCrossAttention(cross_atten_op, c_q_proj, Transformers.Layers.NSplit(static(2), c_kv_proj), c_o_proj)
    return ca
end



struct CustomPostNormResidual{L, N} <: Transformers.Layers.LayerStruct
    layer::L
    norm::N
end
Flux.@functor CustomPostNormResidual

argument_names(b::CustomPostNormResidual) = argument_names(b.layer.layer)
argument_names(b::Transformers.Layers.PostNormResidual) = Transformers.Layers.argument_names(b)

function (postnr::CustomPostNormResidual)(nt::NamedTuple)
    y = Transformers.Layers.apply_on_namedtuple(postnr.layer, nt)
    #hidden_state = y.hidden_state + nt.hidden_state
    hidden_state = y.hidden_state + nt.memory      # use memory for residual connection
    r = Transformers.Layers.return_hidden_state(y, hidden_state)
    return Transformers.Layers.apply_on_namedtuple(postnr.norm, r)
end


struct CustomTransformerDecoderBlock{A, C, F} <: Transformers.Layers.AbstractTransformerBlock
    attention::A
    crossattention::C
    feedforward::F
end
Flux.@functor CustomTransformerDecoderBlock

argument_names(b::CustomTransformerDecoderBlock) = Base.merge_names(
    Base.merge_names(argument_names(b.crossattention), argument_names(b.attention)),
    argument_names(b.feedforward)
)

Transformers.Layers.argument_names(b::CustomTransformerDecoderBlock) = argument_names(b)

(b::CustomTransformerDecoderBlock)(nt::NamedTuple) =
    Transformers.Layers.apply_on_namedtuple(b.feedforward, Transformers.Layers.apply_on_namedtuple(b.crossattention, Transformers.Layers.apply_on_namedtuple(b.attention, nt)))


CustomTransformerDecoderBlock(
    head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = CustomTransformerDecoderBlock(
    gelu, head, hidden_size, head_hidden_size, intermediate_size;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

CustomTransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
) = PostNormTransformerDecoderBlock(
    act, head, hidden_size, head_hidden_size, intermediate_size, true;
    dropout, attention_dropout, cross_attention_dropout, return_score, return_self_attention_score
)

function PostNormTransformerDecoderBlock(
    act, head::Int, hidden_size::Int, head_hidden_size::Int, intermediate_size::Int, UseCustomCrossAttention::Bool=true;
    dropout = nothing, attention_dropout = nothing, cross_attention_dropout = nothing,
    return_score = false, return_self_attention_score = false,
)
    
    sa = Transformers.Layers.SelfAttention(head, hidden_size, head_hidden_size;
                       dropout = attention_dropout, causal = true, return_score = return_self_attention_score)

    if UseCustomCrossAttention
        ca = CustomCrossAttention(head, hidden_size, head_hidden_size; dropout = cross_attention_dropout, return_score)
    else
        ca = Transformers.Layers.CrossAttention(head, hidden_size, head_hidden_size; dropout = cross_attention_dropout, return_score)
    end

    ff1 = Transformers.Layers.Dense(act, hidden_size, intermediate_size)
    ff2 = Transformers.Layers.Dense(intermediate_size, hidden_size)
    return CustomTransformerDecoderBlock(
        Transformers.Layers.PostNormResidual(
            Transformers.Layers.DropoutLayer(sa, dropout),
            Transformers.Layers.LayerNorm(hidden_size)),
        CustomPostNormResidual(
            Transformers.Layers.DropoutLayer(ca, dropout),
            Transformers.Layers.LayerNorm(hidden_size)),
        Transformers.Layers.PostNormResidual(
            Transformers.Layers.DropoutLayer(Chain(ff1, ff2), dropout),
            Transformers.Layers.LayerNorm(hidden_size)))
end




Base.@kwdef struct MATEncoder
    embedding
    position_encoding
    nl
    dropout
    block
    embedding_v
    position_encoding_v
    nl_v
    dropout_v
    block_v
    head
    jointPPO
end

Flux.@functor MATEncoder

function (st::MATEncoder)(x)
    vv = st.embedding_v(x)
    x = st.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ st.position_encoding(1:N) # (dm, N, B)

    x = st.nl(x)

    x = st.dropout(x)                # (dm, N, B)

    x = st.block(x, nothing)     # (dm, N, B)
    rep = x[:hidden_state]

    if st.jointPPO
        vv = vv .+ st.position_encoding_v(1:N)
        vv = st.nl_v(vv)
        vv = st.dropout_v(vv)                # (dm, N, B)
        vv = st.block_v(vv, nothing)     # (dm, N, B)
        vv = vv[:hidden_state]

        sr = size(vv)
        v = st.head( reshape(vv, sr[1]*sr[2], sr[3]) ) 
        v = reshape(v, 1, 1, sr[3])                   # (1, 1, B)
        v = repeat(v, 1,sr[2],1)                   # (1, N, B)
    else
        vv = vv .+ st.position_encoding_v(1:N)
        vv = st.nl_v(vv)
        vv = st.dropout_v(vv)                # (dm, N, B)
        vv = st.block_v(vv, nothing)     # (dm, N, B)
        v = st.head(vv[:hidden_state]) #st.head(rep)                   # (1, N, B)
    end

    rep, v
end


Base.@kwdef struct MATDecoder
    embedding
    position_encoding
    nl
    dropout
    block
    head
    logσ
    logσ_is_network::Bool = false
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
end

Flux.@functor MATDecoder

function (st::MATDecoder)(x, obs_rep)
    x = st.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ st.position_encoding(1:N) # (dm, N, B)

    x = st.nl(x)

    x = st.dropout(x)                # (dm, N, B)

    x = st.block(x, obs_rep, Masks.CausalMask(), Masks.CausalMask())     # (dm, N, B)
    
    x = x[:hidden_state]

    x = st.head(x)                   # (1, N, B)

    if st.logσ_is_network
        raw_logσ = logσ(obs_rep)
    else
        if ndims(x) >= 2
            # TODO: Make it GPU friendly again (CUDA.fill or like Flux Dense Layer does it with bias - Linear Layer with freezing)
            #raw_logσ = hcat([raw_logσ for i in 1:size(μ)[2]]...)
            raw_logσ = send_to_device(device(st.logσ), Float32.(ones(size(x)))) .* st.logσ
        else
            raw_logσ = st.logσ[:]
        end
    end

    logσ = clamp.(raw_logσ, log(st.min_σ), log(st.max_σ))

    return x, logσ
end
















function create_chain_mat(;ns, na, use_gpu, is_actor, init, nna_scale, drop_middle_layer, fun = relu, tanh_end = false)
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

function create_logσ_mat(;logσ_is_network, ns, na, use_gpu, init, nna_scale, drop_middle_layer, fun = relu, start_logσ = 0.0)

    res = nothing

    if logσ_is_network
        res = create_chain_mat(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = false)
    else
        res = Matrix(Matrix(Float32.(ones(na) .* start_logσ)')')
    end

    res = use_gpu ? res |> gpu : res

    return res
end

function create_agent_mat(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = leakyrelu, fun_critic = nothing, n_actors = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = true, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, dim_model = 64, block_num = 1, head_num = 4, head_dim = nothing, ffn_dim = 120, drop_out = 0.1, betas = (0.99, 0.99), jointPPO = false, customCrossAttention = true, one_by_one_training = false, clip_range = 0.2f0)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(head_dim)                 &&  (head_dim = Int(floor(dim_model/head_num)))

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    
    context_size = n_actors

    if jointPPO
        if drop_middle_layer_critic
            head_encoder = Dense(dim_model*n_actors, 1, fun)
        else
            head_encoder = Chain(Dense(dim_model*n_actors, ffn_dim, fun),Dense(ffn_dim, 1, fun))
        end
    else
        if drop_middle_layer_critic
            head_encoder = Dense(dim_model, 1, fun)
        else
            head_encoder = Chain(Dense(dim_model, ffn_dim, fun),Dense(ffn_dim, 1, fun))
        end
    end

    if drop_middle_layer
        head_decoder = Dense(dim_model, na, fun)
    else
        head_decoder = Chain(Dense(dim_model, ffn_dim, fun),Dense(ffn_dim, na, fun))
    end

    if customCrossAttention
        decoder_block = Transformer(CustomTransformerDecoderBlock, block_num, head_num, dim_model, head_dim, ffn_dim; dropout = drop_out)
    else
        decoder_block = Transformer(TransformerDecoderBlock, block_num, head_num, dim_model, head_dim, ffn_dim; dropout = drop_out)
    end

    encoder = MATEncoder(
        embedding = Dense(ns, dim_model, fun, bias = false),
        position_encoding = Embedding(context_size => dim_model),
        nl = LayerNorm(dim_model),
        dropout = Dropout(drop_out),
        block = Transformer(TransformerBlock, block_num, head_num, dim_model, head_dim, ffn_dim; dropout = drop_out),
        embedding_v = Dense(ns, dim_model, fun, bias = false),
        position_encoding_v = Embedding(context_size => dim_model),
        nl_v = LayerNorm(dim_model),
        dropout_v = Dropout(drop_out),
        block_v = Transformer(TransformerBlock, block_num, head_num, dim_model, head_dim, ffn_dim; dropout = drop_out),
        head = head_encoder,
        jointPPO = jointPPO,
    )

    decoder = MATDecoder(
        embedding = Dense(na, dim_model, fun, bias = false),
        position_encoding = Embedding(context_size => dim_model),
        nl = LayerNorm(dim_model),
        dropout = Dropout(drop_out),
        block = decoder_block,
        head = head_decoder,
        logσ = create_logσ_mat(logσ_is_network = logσ_is_network, ns = dim_model, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, fun = fun, start_logσ = start_logσ),
        logσ_is_network = logσ_is_network,
        max_σ = max_σ
    )

    encoder_optimizer = OptimiserChain(ClipNorm(clip_grad), Adam(learning_rate, betas))
    decoder_optimizer = OptimiserChain(ClipNorm(clip_grad), Adam(learning_rate, betas))

    encoder_state_tree = Flux.setup(encoder_optimizer, encoder)
    decoder_state_tree = Flux.setup(decoder_optimizer, decoder)

    Agent(
        policy = MATPolicy(
            encoder = encoder,
            decoder = decoder,
            encoder_optimizer = encoder_optimizer,
            decoder_optimizer = decoder_optimizer,
            encoder_state_tree = encoder_state_tree,
            decoder_state_tree = decoder_state_tree,
            n_actors = n_actors,
            γ = y,
            λ = p,
            clip_range = clip_range,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            rng = rng,
            update_freq = update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            jointPPO = jointPPO,
            one_by_one_training = one_by_one_training,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_actors),
                action = Float32 => (size(action_space)[1], n_actors),
                action_log_prob = Float32 => (n_actors),
                reward = Float32 => (n_actors),
                terminal = Bool => (n_actors,),
                values = Float32 => (1, n_actors),
        ),
    )
end




Base.@kwdef mutable struct MATPolicy <: AbstractPolicy
    encoder
    decoder
    encoder_optimizer
    decoder_optimizer
    encoder_state_tree
    decoder_state_tree
    n_actors::Int = 0
    γ::Float32 = 0.99f0
    λ::Float32 = 0.95f0
    clip_range::Float32 = 0.2f0
    n_microbatches::Int = 4
    n_epochs::Int = 4
    actor_loss_weight::Float32 = 1.0f0
    critic_loss_weight::Float32 = 0.5f0
    entropy_loss_weight::Float32 = 0.0f0
    rng = StableRNG()
    update_freq::Int = 200
    update_step::Int = 0
    clip1::Bool = false
    normalize_advantage::Bool = true
    start_steps = -1
    start_policy = nothing
    target_kl = 100.0
    last_action_log_prob::Vector{Float32} = [0.0]
    next_values::Vector{Float32} = [0.0]
    jointPPO::Bool = false
    one_by_one_training::Bool = false
end


function prob(
    p::MATPolicy,
    state::AbstractArray,
    mask,
)
    na = size(p.decoder.embedding.weight)[2]
    batch_size = length(size(state)) == 3 ? size(state)[3] : 1

    obsrep, val = p.encoder(state)

    μ, logσ = p.decoder(zeros(Float32,na,1,batch_size), obsrep[:,1:1,:])

    for n in 2:p.n_actors
        newμ, newlogσ = p.decoder(cat(zeros(Float32,na,1,batch_size), μ, dims=2), obsrep[:,1:n,:])

        μ = cat(μ, newμ[:,end:end,:], dims=2)
        logσ = cat(logσ, newlogσ[:,end:end,:], dims=2)
    end

    StructArray{Normal}((μ, exp.(logσ)))
end


function prob(p::MATPolicy, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::MATPolicy, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::MATPolicy)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end


function (p::MATPolicy)(env::AbstractEnv)

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

        p.last_action_log_prob = log_p[:]

        action
    end
end

function (agent::Agent{<:MATPolicy})(env::MultiThreadEnv)

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
    ::MATPolicy,
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
    policy::MATPolicy,
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

    obs_rep, values = policy.encoder(send_to_device(device(policy.encoder), env.state)) |> send_to_host

    push!(trajectory[:values], values)
end


function update!(
    trajectory::AbstractTrajectory,
    policy::MATPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)[:]

    next_obs_rep, next_values = policy.encoder(send_to_device(device(policy.encoder), env.state)) |> send_to_host

    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
    policy.next_values = next_values[:]
end

function update!(
    p::MATPolicy,
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





function _update!(p::MATPolicy, t::Any)
    rng = p.rng
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight
    D = device(p.encoder)
    to_device(x) = send_to_device(D, x)

    n_actors, n_rollout = size(t[:terminal])
    @assert n_rollout % n_microbatches == 0 "size mismatch"
    microbatch_size = n_rollout ÷ n_microbatches

    n = length(t)
    states = to_device(t[:state])

    values = reshape(flatten_batch(t[:values]), n_actors, :)
    next_values = cat(values[:,2:end], p.next_values, dims=2)

    rewards = t[:reward]
    terminal = t[:terminal]

    if p.jointPPO
        values = values[1,:]
        next_values = next_values[1,:]
        rewards = rewards[1,:]
        terminal = terminal[1,:]
    end

    advantages = generalized_advantage_estimation(
        rewards,
        values,
        next_values,
        γ,
        λ;
        dims=2,
        terminal=terminal
    )

    if p.jointPPO
        returns = to_device(advantages .+ values[1:n_rollout])
    else
        returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    end
    advantages = to_device(advantages)

    actions = to_device(t[:action])
    action_log_probs = t[:action_log_prob]

    if p.jointPPO
        action_log_probs = sum(action_log_probs, dims=1)[:]
    else
        action_log_probs = reshape(action_log_probs, 1, size(action_log_probs)[1], size(action_log_probs)[2])
    end

    stop_update = false

    if p.one_by_one_training
        n_epochs = n_epochs * p.n_actors
    end

    rand_actor_inds = shuffle!(rng, Vector(1:p.n_actors))
    reverse_actor_inds = Vector(p.n_actors:-1:1)

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            #global s, a, r, log_p, adv

            s = to_device(collect(select_last_dim(states, inds)))
            a = to_device(collect(select_last_dim(actions, inds)))

            r = select_last_dim(returns, inds)
            log_p = select_last_dim(action_log_probs, inds)
            adv = select_last_dim(advantages, inds)

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-5 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end


            g_encoder, g_decoder = Flux.gradient(p.encoder, p.decoder) do encoder, decoder
                #global obs_rep, v′, temp_act, μ, logσ, log_p′ₐ, ratio

                obs_rep, v′ = encoder(s)
                
                # obs_rep, v′_no = p.encoder(s)
                # obs_rep_no, v′ = encoder(s)

                #parallel act
                temp_act = cat(zeros(Float32,1,1,size(a)[3]),a[:,1:end-1,:],dims=2)
                μ, logσ = decoder(temp_act, obs_rep)
                
                log_p′ₐ = sum(normlogpdf(μ, exp.(logσ), a), dims=1)

                
                if p.jointPPO
                    log_p′ₐ = sum(log_p′ₐ, dims=2)[:]
                end

                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl && (i > 1 || epoch > 1) # only in second batch
                        println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        stop_update = true
                    end
                end

                #adv = reshape(adv, 1, size(adv)[1], size(adv)[2])
                #r = reshape(r, 1, size(r)[1], size(r)[2])


                

                if p.jointPPO
                    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

                    surr1 = ratio .* adv
                    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))

                    critic_loss = mean((r .- v′[1,1,:]) .^ 2)

                elseif p.one_by_one_training
                    temp_index = 1 + epoch%p.n_actors
                    #actor_index = rand_actor_inds[temp_index]
                    #actor_index = reverse_actor_inds[temp_index]
                    actor_index = temp_index

                    entropy_loss = mean(size(logσ[:,actor_index,:], 1) * (log(2.0f0π) + 1) .+ sum(logσ[:,actor_index,:]; dims=1)) / 2

                    surr1 = ratio[1,actor_index,:] .* adv[actor_index,:]
                    surr2 = clamp.(ratio[1,actor_index,:], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv[actor_index,:]

                    actor_loss = -mean(min.(surr1, surr2))

                    critic_loss = mean((r[actor_index,:] .- v′[1,actor_index,:]) .^ 2)

                else
                    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

                    surr1 = ratio[1,:,:] .* adv
                    surr2 = clamp.(ratio[1,:,:], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))

                    critic_loss = mean((r .- v′[1,:,:]) .^ 2)
                end


                
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                loss
            end
            
            if !stop_update
                Flux.update!(p.encoder_state_tree, p.encoder, g_encoder)
                Flux.update!(p.decoder_state_tree, p.decoder, g_decoder)
            else
                break
            end

        end

        if stop_update
            break
        end
    end
end

