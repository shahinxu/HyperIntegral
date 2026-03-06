using DelimitedFiles, JSON, Statistics, Combinatorics

include("this.jl")

function tupset(rows::Vector{Any})
    s = Set{Tuple{Vararg{Int}}}()
    for r in rows
        rr = sort(Int.(r))
        push!(s, Tuple(rr))
    end
    return s
end

function inferred_scores(Ainf::Dict{Int64,Matrix{Float64}}, order::Int)
    d = Dict{Tuple{Vararg{Int}},Float64}()
    if !haskey(Ainf, order)
        return d
    end
    A = Ainf[order]
    if size(A,1) == 0
        return d
    end
    for i in 1:size(A,1)
        nodes = sort(Int.(A[i,1:order]))
        if length(unique(nodes)) == order
            key = Tuple(nodes)
            score = abs(Float64(A[i,order+1]))
            d[key] = max(get(d, key, 0.0), score)
        end
    end
    return d
end

function inferred_hyperedges_from_scores(score_map::Dict{Tuple{Vararg{Int}},Float64}, threshold::Float64)
    s = Set{Tuple{Vararg{Int}}}()
    for (k, v) in score_map
        if v >= threshold
            push!(s, k)
        end
    end
    return s
end

function auc_roc(y::Vector{Int}, s::Vector{Float64})
    n = length(y)
    pos = sum(y)
    neg = n - pos
    if pos == 0 || neg == 0
        return NaN
    end

    idx = sortperm(s)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && s[idx[j+1]] == s[idx[i]]
            j += 1
        end
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[idx[k]] = avg_rank
        end
        i = j + 1
    end

    sum_ranks_pos = 0.0
    for i in 1:n
        if y[i] == 1
            sum_ranks_pos += ranks[i]
        end
    end

    return (sum_ranks_pos - pos * (pos + 1) / 2) / (pos * neg)
end

function prf(inf::Set{T}, tru::Set{T}) where T
    tp = length(intersect(inf, tru))
    fp = length(setdiff(inf, tru))
    fn = length(setdiff(tru, inf))
    prec = tp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp == 0 ? 0.0 : tp / (tp + fn)
    f1 = (prec + rec) == 0 ? 0.0 : 2 * prec * rec / (prec + rec)
    return tp, fp, fn, prec, rec, f1
end

function append_split_metrics!(lines::Vector{String}, split::String, Ainf::Dict{Int64,Matrix{Float64}}, truth, n_nodes::Int, ooi::Vector{Int}, bin_thresh::Float64)
    order_key = Dict(2 => "edges", 3 => "triangles", 4 => "quads", 5 => "quints")
    for o in ooi
        if !haskey(order_key, o)
            continue
        end
        key = order_key[o]
        tru_rows = get(truth, key, Any[])
        tru = tupset(tru_rows)

        score_map = inferred_scores(Ainf, o)
        inf = inferred_hyperedges_from_scores(score_map, bin_thresh)
        tp, fp, fn, prec, rec, f1 = prf(inf, tru)

        all_edges = [Tuple(c) for c in combinations(1:n_nodes, o)]
        y_true = [e in tru ? 1 : 0 for e in all_edges]
        y_score = [get(score_map, e, 0.0) for e in all_edges]
        auc = auc_roc(y_true, y_score)
        auc_str = isnan(auc) ? "NaN" : string(round(auc, digits=4))

        push!(lines, "$split: order=$o, true=$(length(tru)), inferred=$(length(inf)), tp=$tp, fp=$fp, fn=$fn, precision=$(round(prec,digits=4)), recall=$(round(rec,digits=4)), f1=$(round(f1,digits=4)), auc=$auc_str")
    end
end

function main()
    n_nodes = parse(Int, get(ENV, "SC_NODES", "8"))
    max_order = parse(Int, get(ENV, "SC_ORDER", "3"))
    n_samples = parse(Int, get(ENV, "SC_SAMPLES", "300"))
    noise = parse(Float64, get(ENV, "SC_NOISE", "0.0"))
    graph_seed = parse(Int, get(ENV, "SC_GRAPH_SEED", "42"))
    train_seed = parse(Int, get(ENV, "SC_TRAIN_SEED", "123"))
    test_seed = parse(Int, get(ENV, "SC_TEST_SEED", "456"))

    λ = parse(Float64, get(ENV, "SC_LAMBDA", "0.05"))
    ρ = parse(Float64, get(ENV, "SC_RHO", "0.001"))
    niter = parse(Int, get(ENV, "SC_NITER", "10"))
    bin_thresh = parse(Float64, get(ENV, "SC_BIN_THRESH", "1e-4"))

    data_dir = joinpath(@__DIR__, "social-data")
    mkpath(data_dir)

    py = get(ENV, "SC_PY", "python")
    cmd = `$py $(joinpath(@__DIR__, "social_export.py")) --n_nodes $n_nodes --max_order $max_order --n_samples $n_samples --noise $noise --graph_seed $graph_seed --train_seed $train_seed --test_seed $test_seed --out_dir $data_dir`
    println("[THIS social] exporting data via: ", cmd)
    run(cmd)

    X_train = readdlm(joinpath(data_dir, "X_train.csv"), ',', Float64)
    Y_train = readdlm(joinpath(data_dir, "Y_train.csv"), ',', Float64)
    X_test = readdlm(joinpath(data_dir, "X_test.csv"), ',', Float64)
    Y_test = readdlm(joinpath(data_dir, "Y_test.csv"), ',', Float64)
    truth = JSON.parsefile(joinpath(data_dir, "truth.json"))

    ooi = collect(2:max_order)
    dmax = max_order

    println("[THIS social] running THIS(train) with n=$(size(X_train,1)), T=$(size(X_train,2)), dmax=$dmax, lambda=$λ, rho=$ρ")
    Ainf_train, coeff_train, relerr_train = this(X_train, Y_train, ooi, dmax, λ, ρ, niter)

    println("[THIS social] running THIS(test) with n=$(size(X_test,1)), T=$(size(X_test,2)), dmax=$dmax, lambda=$λ, rho=$ρ")
    Ainf_test, coeff_test, relerr_test = this(X_test, Y_test, ooi, dmax, λ, ρ, niter)

    lines = String[]
    push!(lines, "scene=social_contagion")
    push!(lines, "protocol=strict_observed_fdiff_independent_test")
    push!(lines, "n_nodes=$n_nodes, max_order=$max_order, n_samples=$n_samples, noise=$noise")
    push!(lines, "graph_seed=$graph_seed, train_seed=$train_seed, test_seed=$test_seed")
    push!(lines, "lambda=$λ, rho=$ρ, niter=$niter")
    push!(lines, "binary_score_threshold=$bin_thresh")
    push!(lines, "relerr_train=$relerr_train")
    push!(lines, "relerr_test=$relerr_test")
    push!(lines, "")

    append_split_metrics!(lines, "train", Ainf_train, truth, n_nodes, ooi, bin_thresh)
    append_split_metrics!(lines, "test", Ainf_test, truth, n_nodes, ooi, bin_thresh)

    out_path = joinpath(data_dir, "this_social_metrics.txt")
    open(out_path, "w") do io
        for ln in lines
            println(io, ln)
        end
    end

    println("[THIS social] metrics saved to: ", out_path)
end

main()
