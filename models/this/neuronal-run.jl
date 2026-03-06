using DelimitedFiles, JSON, Statistics, Combinatorics

include("this.jl")

"""
Run THIS on neuronal synchronization data exported from HyperPINN/lib_neuronal_synchronization.

Usage in Julia REPL from THIS dir:
    include("neuronal-run.jl")

Optional env vars:
    NS_NODES, NS_ORDER, NS_SAMPLES, NS_NOISE, NS_LAMBDA, NS_RHO, NS_NITER, NS_PY
"""

function tupset(rows::Vector{Any})
    s = Set{Tuple{Vararg{Int}}}()
    for r in rows
        rr = sort(Int.(r))
        push!(s, Tuple(rr))
    end
    return s
end

function inferred_hyperedges(Ainf::Dict{Int64,Matrix{Float64}}, order::Int)
    if !haskey(Ainf, order)
        return Set{Tuple{Vararg{Int}}}()
    end
    A = Ainf[order]
    if size(A,1) == 0
        return Set{Tuple{Vararg{Int}}}()
    end
    s = Set{Tuple{Vararg{Int}}}()
    for i in 1:size(A,1)
        nodes = sort(Int.(A[i,1:order]))
        if length(unique(nodes)) == order
            push!(s, Tuple(nodes))
        end
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

function main()
    n_nodes = parse(Int, get(ENV, "NS_NODES", "9"))
    max_order = parse(Int, get(ENV, "NS_ORDER", "5"))
    n_samples = parse(Int, get(ENV, "NS_SAMPLES", "300"))
    noise = parse(Float64, get(ENV, "NS_NOISE", "0.0"))

    λ = parse(Float64, get(ENV, "NS_LAMBDA", "0.05"))
    ρ = parse(Float64, get(ENV, "NS_RHO", "0.001"))
    niter = parse(Int, get(ENV, "NS_NITER", "10"))
    bin_thresh = parse(Float64, get(ENV, "NS_BIN_THRESH", "0.01"))

    data_dir = joinpath(@__DIR__, "neuronal-data")
    mkpath(data_dir)

    py = get(ENV, "NS_PY", "python")
    cmd = `$py $(joinpath(@__DIR__, "neuronal_export.py")) --n_nodes $n_nodes --max_order $max_order --n_samples $n_samples --noise $noise --out_dir $data_dir`
    println("[THIS neuronal] exporting data via: ", cmd)
    run(cmd)

    X = readdlm(joinpath(data_dir, "X.csv"), ',', Float64)
    Y = readdlm(joinpath(data_dir, "Y.csv"), ',', Float64)
    truth = JSON.parsefile(joinpath(data_dir, "truth.json"))

    ooi = collect(2:max_order)
    dmax = max_order

    println("[THIS neuronal] running THIS with n=$(size(X,1)), T=$(size(X,2)), dmax=$dmax, lambda=$λ, rho=$ρ")
    Ainf, coeff, relerr = this(X, Y, ooi, dmax, λ, ρ, niter)

    order_key = Dict(2 => "edges", 3 => "triangles", 4 => "quads", 5 => "quints")
    lines = String[]
    push!(lines, "scene=neuronal_synchronization")
    push!(lines, "n_nodes=$n_nodes, max_order=$max_order, n_samples=$n_samples, noise=$noise")
    push!(lines, "lambda=$λ, rho=$ρ, niter=$niter")
    push!(lines, "binary_score_threshold=$bin_thresh")
    push!(lines, "relerr=$relerr")
    push!(lines, "")

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

        push!(lines, "order=$o, true=$(length(tru)), inferred=$(length(inf)), tp=$tp, fp=$fp, fn=$fn, precision=$(round(prec,digits=4)), recall=$(round(rec,digits=4)), f1=$(round(f1,digits=4)), auc=$auc_str")
    end

    out_path = joinpath(data_dir, "this_neuronal_metrics.txt")
    open(out_path, "w") do io
        for ln in lines
            println(io, ln)
        end
    end

    println("[THIS neuronal] metrics saved to: ", out_path)
end

main()
