"""
Run THIS method multiple times for different M values
Each M value is tested with 20 different random initial conditions
"""

include("THIS/this.jl")

using LinearAlgebra, Statistics, DelimitedFiles
using Printf
using Combinatorics
using Random
using Dates

# Include all functions from run_this_rossler.jl
include("run_this_rossler.jl")

# Modified main function that accepts M as parameter
function run_single_experiment(M::Int, N::Int, tmax::Float64, max_order::Int)
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
    
    X_x = X_full[1:N, :]
    X_y = X_full[N+1:2*N, :]
    X_z = X_full[2*N+1:3*N, :]
    
    Y_x = Y_full[1:N, :]
    Y_y = Y_full[N+1:2*N, :]
    Y_z = Y_full[2*N+1:3*N, :]
    
    # Normalize
    X_x = X_x ./ mean(abs.(X_x))
    X_y = X_y ./ mean(abs.(X_y))
    X_z = X_z ./ mean(abs.(X_z))
    Y_x = Y_x ./ mean(abs.(Y_x))
    Y_y = Y_y ./ mean(abs.(Y_y))
    Y_z = Y_z ./ mean(abs.(Y_z))
    
    # THIS parameters
    λ = 0.1
    ρ = 1.0
    niter = 10
    dmax = max_order
    ooi = collect(2:max_order)
    
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
        all_possible = generate_all_combinations(N, order)
        true_edges = true_edges_dict[order]
        inferred_edges = extract_hyperedges_from_THIS(Ainf_combined, N, order, threshold)
        
        y_true = Float64[]
        y_score = Float64[]
        
        for edge in all_possible
            sorted_edge = sort(edge)
            push!(y_true, sorted_edge in true_edges ? 1.0 : 0.0)
            
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
        
        auc_val = compute_auc(y_true, y_score)
        auc_scores[order] = auc_val
    end
    
    return auc_scores
end

# Main loop
function run_multiple_experiments()
    N = 8
    tmax = 20.0
    max_order = 7
    
    M_values = [300, 200, 100, 50, 20, 10]
    n_runs = 20
    
    # Create main results directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    results_dir = "results_THIS_multiple/$timestamp"
    mkpath(results_dir)
    
    # Store all results
    all_results = Dict()
    
    for M in M_values
        println("\n" * "="^80)
        println("Running M=$M (20 trials)")
        println("="^80)
        
        M_results = []
        
        # Create directory and files for this M value
        M_dir = "$results_dir/M_$M"
        mkpath(M_dir)
        
        # Initialize files
        all_trials_file = "$M_dir/all_trials.txt"
        open(all_trials_file, "w") do f
            write(f, "M=$M, N=$N, max_order=$max_order, tmax=$tmax\n")
            write(f, "Number of trials: $n_runs\n\n")
        end
        
        for trial in 1:n_runs
            print("\r  Trial $trial/20...")
            flush(stdout)
            
            auc_scores = run_single_experiment(M, N, tmax, max_order)
            push!(M_results, auc_scores)
            
            # Immediately write this trial's result
            open(all_trials_file, "a") do f
                write(f, "Trial $trial:\n")
                for order in 2:max_order
                    write(f, @sprintf("  %d-edges: %.4f\n", order, auc_scores[order]))
                end
                write(f, "\n")
            end
        end
        println(" Done!")
        
        all_results[M] = M_results
        
        # Compute statistics
        stats = Dict()
        for order in 2:max_order
            values = [M_results[trial][order] for trial in 1:n_runs]
            stats[order] = Dict(
                "mean" => mean(values),
                "std" => std(values),
                "min" => minimum(values),
                "max" => maximum(values)
            )
        end
        
        # Save statistics
        open("$M_dir/statistics.txt", "w") do f
            write(f, "M=$M Statistics (n=$n_runs)\n\n")
            write(f, "Order | Mean    | Std     | Min     | Max\n")
            write(f, "------+---------+---------+---------+---------\n")
            for order in 2:max_order
                write(f, @sprintf("%5d | %.4f | %.4f | %.4f | %.4f\n",
                    order, stats[order]["mean"], stats[order]["std"],
                    stats[order]["min"], stats[order]["max"]))
            end
        end
        
        # Print statistics
        println("\n  Statistics for M=$M:")
        println("  Order | Mean    | Std")
        println("  ------+---------+---------")
        for order in 2:max_order
            println(@sprintf("  %5d | %.4f | %.4f", order, stats[order]["mean"], stats[order]["std"]))
        end
    end
    
    # Save summary across all M values
    open("$results_dir/summary.txt", "w") do f
        write(f, "Summary of Multiple Experiments\n")
        write(f, "N=$N, max_order=$max_order, tmax=$tmax, n_runs=$n_runs\n")
        write(f, "="^80 * "\n\n")
        
        for order in 2:max_order
            write(f, "Order $order AUC:\n")
            write(f, "M    | Mean    | Std     | Min     | Max\n")
            write(f, "-----+---------+---------+---------+---------\n")
            for M in M_values
                M_results = all_results[M]
                values = [M_results[trial][order] for trial in 1:n_runs]
                write(f, @sprintf("%4d | %.4f | %.4f | %.4f | %.4f\n",
                    M, mean(values), std(values), minimum(values), maximum(values)))
            end
            write(f, "\n")
        end
    end
    
    println("\n" * "="^80)
    println("All experiments completed!")
    println("Results saved to: $results_dir/")
    println("="^80)
    
    return all_results, results_dir
end

# Run experiments
all_results, results_dir = run_multiple_experiments()
