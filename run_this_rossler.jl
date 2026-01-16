"""
Run THIS method on Rossler Oscillators with identical setup to Rossler_Oscillators.py
"""

include("THIS/this.jl")

using LinearAlgebra, Statistics, DelimitedFiles
using Printf
using Combinatorics
using Random
using Dates

# ODE solver (simple RK4)
function rk4_step(f, t, y, dt, args)
    k1 = f(t, y, args...)
    k2 = f(t + dt/2, y + dt*k1/2, args...)
    k3 = f(t + dt/2, y + dt*k2/2, args...)
    k4 = f(t + dt, y + dt*k3, args...)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
end

function solve_ode(f, tspan, y0, dt, args)
    t_start, t_end = tspan
    t_vals = collect(t_start:dt:t_end)
    n = length(y0)
    T = length(t_vals)
    
    Y = zeros(n, T)
    Y[:, 1] = y0
    
    for i in 1:T-1
        Y[:, i+1] = rk4_step(f, t_vals[i], Y[:, i], dt, args)
    end
    
    return t_vals, Y
end

# Rossler HOI dynamics - IDENTICAL to Rossler_Oscillators.py
function roessler_hoi(t, x, EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList)
    m1 = length(x)
    N = div(m1, 3)
    
    xold = x[1:N]
    yold = x[N+1:2*N]
    zold = x[2*N+1:3*N]
    
    ar, br, cr = 0.2, 0.2, 0.7
    k, kD = 0.4, 0.3
    
    coup_rete = zeros(N)
    coup_simplicial = zeros(N)
    coup_quads = zeros(N)
    coup_quints = zeros(N)
    coup_sexts = zeros(N)
    coup_septs = zeros(N)
    
    # 2-edges
    for ii in 1:size(EdgeList, 1)
        i1 = EdgeList[ii, 1]
        i2 = EdgeList[ii, 2]
        coup_rete[i1] += xold[i2] - xold[i1]
        coup_rete[i2] += xold[i1] - xold[i2]
    end
    
    # 3-edges (triangles)
    for ii in 1:size(TriangleList, 1)
        i1 = TriangleList[ii, 1]
        i2 = TriangleList[ii, 2]
        i3 = TriangleList[ii, 3]
        coup_simplicial[i1] += xold[i2]^2 * xold[i3] - xold[i1]^3 + xold[i2] * xold[i3]^2 - xold[i1]^3
        coup_simplicial[i2] += xold[i1]^2 * xold[i3] - xold[i2]^3 + xold[i1] * xold[i3]^2 - xold[i2]^3
        coup_simplicial[i3] += xold[i1]^2 * xold[i2] - xold[i3]^3 + xold[i1] * xold[i2]^2 - xold[i3]^3
    end
    
    # 4-edges (quads)
    for ii in 1:size(QuadList, 1)
        i1 = QuadList[ii, 1]
        i2 = QuadList[ii, 2]
        i3 = QuadList[ii, 3]
        i4 = QuadList[ii, 4]
        coup_quads[i1] += xold[i2]^2 * xold[i3] * xold[i4] - xold[i1]^3
        coup_quads[i2] += xold[i1]^2 * xold[i3] * xold[i4] - xold[i2]^3
        coup_quads[i3] += xold[i1]^2 * xold[i2] * xold[i4] - xold[i3]^3
        coup_quads[i4] += xold[i1]^2 * xold[i2] * xold[i3] - xold[i4]^3
    end
    
    # 5-edges (quints)
    for ii in 1:size(QuintList, 1)
        i1 = QuintList[ii, 1]
        i2 = QuintList[ii, 2]
        i3 = QuintList[ii, 3]
        i4 = QuintList[ii, 4]
        i5 = QuintList[ii, 5]
        coup_quints[i1] += yold[i2]^2 * yold[i3] * yold[i4] * yold[i5] - yold[i1]^3
        coup_quints[i2] += yold[i1]^2 * yold[i3] * yold[i4] * yold[i5] - yold[i2]^3
        coup_quints[i3] += yold[i1]^2 * yold[i2] * yold[i4] * yold[i5] - yold[i3]^3
        coup_quints[i4] += yold[i1]^2 * yold[i2] * yold[i3] * yold[i5] - yold[i4]^3
        coup_quints[i5] += yold[i1]^2 * yold[i2] * yold[i3] * yold[i4] - yold[i5]^3
    end
    
    # 6-edges (sexts)
    for ii in 1:size(SextList, 1)
        i1 = SextList[ii, 1]
        i2 = SextList[ii, 2]
        i3 = SextList[ii, 3]
        i4 = SextList[ii, 4]
        i5 = SextList[ii, 5]
        i6 = SextList[ii, 6]
        coup_sexts[i1] += yold[i2]^2 * yold[i3] * yold[i4] * yold[i5] * yold[i6] - yold[i1]^3
        coup_sexts[i2] += yold[i1]^2 * yold[i3] * yold[i4] * yold[i5] * yold[i6] - yold[i2]^3
        coup_sexts[i3] += yold[i1]^2 * yold[i2] * yold[i4] * yold[i5] * yold[i6] - yold[i3]^3
        coup_sexts[i4] += yold[i1]^2 * yold[i2] * yold[i3] * yold[i5] * yold[i6] - yold[i4]^3
        coup_sexts[i5] += yold[i1]^2 * yold[i2] * yold[i3] * yold[i4] * yold[i6] - yold[i5]^3
        coup_sexts[i6] += yold[i1]^2 * yold[i2] * yold[i3] * yold[i4] * yold[i5] - yold[i6]^3
    end
    
    # 7-edges (septs)
    for ii in 1:size(SeptList, 1)
        i1 = SeptList[ii, 1]
        i2 = SeptList[ii, 2]
        i3 = SeptList[ii, 3]
        i4 = SeptList[ii, 4]
        i5 = SeptList[ii, 5]
        i6 = SeptList[ii, 6]
        i7 = SeptList[ii, 7]
        coup_septs[i1] += zold[i2]^2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i1]^3
        coup_septs[i2] += zold[i1]^2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i2]^3
        coup_septs[i3] += zold[i1]^2 * zold[i2] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i3]^3
        coup_septs[i4] += zold[i1]^2 * zold[i2] * zold[i3] * zold[i5] * zold[i6] * zold[i7] - zold[i4]^3
        coup_septs[i5] += zold[i1]^2 * zold[i2] * zold[i3] * zold[i4] * zold[i6] * zold[i7] - zold[i5]^3
        coup_septs[i6] += zold[i1]^2 * zold[i2] * zold[i3] * zold[i4] * zold[i5] * zold[i7] - zold[i6]^3
        coup_septs[i7] += zold[i1]^2 * zold[i2] * zold[i3] * zold[i4] * zold[i5] * zold[i6] - zold[i7]^3
    end
    
    dxdt1 = -yold .- zold .+ k .* coup_rete .+ kD .* coup_simplicial .+ kD .* coup_quads
    dydt1 = xold .+ ar .* yold .+ kD .* coup_quints .+ kD .* coup_sexts
    dzdt1 = br .+ zold .* (xold .- cr) .+ kD .* coup_septs
    
    return vcat(dxdt1, dydt1, dzdt1)
end

# Get hyperedge configuration - IDENTICAL to Python
function get_hyperedge_config(N)
    configs = Dict(
        8 => Dict(
            "edges" => [[1, 2],[2, 3],[3, 4],[5, 6],[6, 7],[7, 8]],
            "triangles" => [[1, 2, 3],[2, 4, 5],[5, 6, 7],[6, 7, 8]],
            "quads" => [[1, 2, 3, 4]],
            "quints" => [[4, 5, 6, 7, 8]],
            "sexts" => [[1, 2, 3, 4, 5, 6]],
            "septs" => [[1, 2, 4, 5, 6, 7, 8]]
        )
    )
    
    if !haskey(configs, N)
        error("N must be 8, got $N")
    end
    
    config = configs[N]
    
    # Convert to matrices
    EdgeList = isempty(config["edges"]) ? zeros(Int, 0, 2) : hcat([e for e in config["edges"]]...)'
    TriangleList = isempty(config["triangles"]) ? zeros(Int, 0, 3) : hcat([e for e in config["triangles"]]...)'
    QuadList = isempty(config["quads"]) ? zeros(Int, 0, 4) : hcat([e for e in config["quads"]]...)'
    QuintList = isempty(config["quints"]) ? zeros(Int, 0, 5) : hcat([e for e in config["quints"]]...)'
    SextList = isempty(config["sexts"]) ? zeros(Int, 0, 6) : hcat([e for e in config["sexts"]]...)'
    SeptList = isempty(config["septs"]) ? zeros(Int, 0, 7) : hcat([e for e in config["septs"]]...)'
    
    return EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList
end

# Generate all possible combinations
function generate_all_combinations(N, order)
    return [collect(c) for c in combinations(1:N, order)]
end

# Compute AUC
function compute_auc(y_true, y_score)
    # Sort by score (descending)
    idx = sortperm(y_score, rev=true)
    y_true_sorted = y_true[idx]
    
    n_pos = sum(y_true)
    n_neg = length(y_true) - n_pos
    
    if n_pos == 0 || n_neg == 0
        return NaN
    end
    
    # Compute TPR and FPR
    tp = 0.0
    fp = 0.0
    auc_val = 0.0
    prev_fp = 0.0
    
    for i in 1:length(y_true_sorted)
        if y_true_sorted[i] == 1
            tp += 1
        else
            fp += 1
            # Add area of trapezoid
            auc_val += tp / n_pos * (fp - prev_fp) / n_neg
            prev_fp = fp
        end
    end
    
    return auc_val
end

# Extract hyperedges from THIS inference results
function extract_hyperedges_from_THIS(Ainf, N, order, threshold=1e-3)
    inferred_edges = []
    
    if !haskey(Ainf, order)
        return inferred_edges
    end
    
    A_matrix = Ainf[order]
    if size(A_matrix, 1) == 0
        return inferred_edges
    end
    for i in 1:size(A_matrix, 1)
        coef = A_matrix[i, end]
        if abs(coef) > threshold
            agent_idx = Int(A_matrix[i, 1])
            other_nodes = [Int(A_matrix[i, j]) for j in 2:order]
            nodes = sort(vcat([agent_idx], other_nodes))
            if !(nodes in inferred_edges)
                push!(inferred_edges, nodes)
            end
        end
    end
    
    return inferred_edges
end

# Main execution
function main()    
    N = 8
    M = 300
    tmax = 20.0
    max_order = 7
    
    EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList = get_hyperedge_config(N)
    
    x0 = rand(3 * N) .* 2 .- 1
    dt = tmax / M
    t_vals, X_full = solve_ode(roessler_hoi, (0.0, tmax), x0, dt, 
                                (EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList))
    
    Y_full = zeros(size(X_full))
    for i in 1:length(t_vals)
        Y_full[:, i] = roessler_hoi(t_vals[i], X_full[:, i], 
                                     EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList)
    end
    
    X_x = X_full[1:N, :]           # x coordinates
    X_y = X_full[N+1:2*N, :]       # y coordinates  
    X_z = X_full[2*N+1:3*N, :]     # z coordinates
    
    Y_x = Y_full[1:N, :]
    Y_y = Y_full[N+1:2*N, :]
    Y_z = Y_full[2*N+1:3*N, :]
    
    # Normalize data so all orders are of comparable magnitudes (following THIS original approach)
    X_x = X_x ./ mean(abs.(X_x))
    X_y = X_y ./ mean(abs.(X_y))
    X_z = X_z ./ mean(abs.(X_z))
    Y_x = Y_x ./ mean(abs.(Y_x))
    Y_y = Y_y ./ mean(abs.(Y_y))
    Y_z = Y_z ./ mean(abs.(Y_z))
    
    # THIS default parameters (matching original implementation)
    λ = 0.1   # SINDy threshold
    ρ = 1.0   # Regularization parameter
    niter = 10  # Number of iterations
    dmax = max_order
    ooi = collect(2:max_order)
    
    println("Running THIS (λ=$λ, ρ=$ρ, niter=$niter)...")
    
    Ainf_x, coeff_x, relerr_x = this(X_x, Y_x, ooi, dmax, λ, ρ, niter)
    
    Ainf_y, coeff_y, relerr_y = this(X_y, Y_y, ooi, dmax, λ, ρ, niter)
    
    Ainf_z, coeff_z, relerr_z = this(X_z, Y_z, ooi, dmax, λ, ρ, niter)
    
    Ainf_combined = Dict{Int64,Matrix{Float64}}()
    
    for order in [2, 3, 4]
        if haskey(Ainf_x, order)
            Ainf_combined[order] = Ainf_x[order]
        end
    end
    
    for order in [5, 6]
        if haskey(Ainf_y, order)
            Ainf_combined[order] = Ainf_y[order]
        end
    end
    
    # Order 7 comes from z-dynamics
    if haskey(Ainf_z, 7)
        Ainf_combined[7] = Ainf_z[7]
    end
    
    threshold = 0.01
    true_edges_dict = Dict(
        2 => Set([sort(e) for e in eachrow(EdgeList)]),
        3 => Set([sort(e) for e in eachrow(TriangleList)]),
        4 => Set([sort(e) for e in eachrow(QuadList)]),
        5 => Set([sort(e) for e in eachrow(QuintList)]),
        6 => Set([sort(e) for e in eachrow(SextList)]),
        7 => Set([sort(e) for e in eachrow(SeptList)])
    )
    
    auc_scores = Dict{Int64, Float64}()
    
    for order in 2:max_order
        # Generate all possible hyperedges
        all_possible = generate_all_combinations(N, order)
        
        # Get ground truth
        true_edges = true_edges_dict[order]
        
        # Extract inferred edges
        inferred_edges = extract_hyperedges_from_THIS(Ainf_combined, order, threshold)
        inferred_set = Set([sort(e) for e in inferred_edges])
        
        # Compute scores for AUC
        y_true = Float64[]
        y_score = Float64[]
        
        for edge in all_possible
            sorted_edge = sort(edge)
            # True label
            push!(y_true, sorted_edge in true_edges ? 1.0 : 0.0)
            
            # Score: use coefficient magnitude if edge is in Ainf, else 0
            score = 0.0
            if haskey(Ainf_combined, order)
                A_matrix = Ainf_combined[order]
                for i in 1:size(A_matrix, 1)
                    agent_idx = Int(A_matrix[i, 1])
                    other_nodes = [Int(A_matrix[i, j]) for j in 2:order]
                    nodes = sort(vcat([agent_idx], other_nodes))
                    if nodes == sorted_edge
                        score = max(score, abs(A_matrix[i, end]))
                    end
                end
            end
            push!(y_score, score)
        end
        
        # Compute AUC
        auc_val = compute_auc(y_true, y_score)
        auc_scores[order] = auc_val
    end
    
    # Summary
    println("\nAUC Results:")
    for order in 2:max_order
        println(@sprintf("  %d-edges: %.4f", order, auc_scores[order]))
    end
    
    # Save results
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    results_dir = "results_THIS/$timestamp"
    mkpath(results_dir)
    
    # Save AUC scores
    open("$results_dir/auc_scores.txt", "w") do f
        write(f, "THIS Method on Rossler Oscillators\n")
        write(f, "N=$N, max_order=$max_order, M=$M, tmax=$tmax\n")
        write(f, "λ=$λ, ρ=$ρ, threshold=$threshold\n\n")
        write(f, "AUC Scores:\n")
        for order in 2:max_order
            write(f, @sprintf("  %d-edges: %.4f\n", order, auc_scores[order]))
        end
    end
    
    # Save detailed inference results for debugging
    open("$results_dir/inference_details.txt", "w") do f
        write(f, "Detailed Inference Results\n")
        write(f, "="^80 * "\n\n")
        for order in 2:max_order
            write(f, "Order $order:\n")
            if haskey(Ainf_combined, order)
                A_matrix = Ainf_combined[order]
                write(f, "  Matrix shape: $(size(A_matrix))\n")
                write(f, "  Non-zero coefficients:\n")
                for i in 1:size(A_matrix, 1)
                    coef = A_matrix[i, end]
                    if abs(coef) > 1e-6
                        agent_idx = Int(A_matrix[i, 1])
                        other_nodes = [Int(A_matrix[i, j]) for j in 2:order]
                        full_edge = sort(vcat([agent_idx], other_nodes))
                        write(f, "    Agent $agent_idx, others $other_nodes -> hyperedge $full_edge, coef: $coef\n")
                    end
                end
            else
                write(f, "  No results for this order\n")
            end
            write(f, "\n")
        end
    end
    
    println("\nResults -> $results_dir/")
    
    return auc_scores, Ainf_combined
end

# Run main only if this is the main script (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    auc_scores, Ainf = main()
end
