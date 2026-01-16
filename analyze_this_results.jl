"""
Analyze results from run_this_multiple.jl
Calculate mean and variance for each M and order
"""

using Statistics
using Printf

function parse_trial_results(filepath)
    """Parse all_trials.txt file and extract AUC scores"""
    results = Dict{Int, Vector{Float64}}()
    
    for order in 2:7
        results[order] = Float64[]
    end
    
    current_trial = nothing
    open(filepath, "r") do f
        for line in eachline(f)
            if startswith(line, "Trial")
                current_trial = true
            elseif occursin("-edges:", line)
                # Parse line like "  2-edges: 0.1439"
                parts = split(line, ":")
                order_part = strip(parts[1])
                order = parse(Int, split(order_part, "-")[1])
                auc = parse(Float64, strip(parts[2]))
                push!(results[order], auc)
            end
        end
    end
    
    return results
end

function analyze_results(results_dir)
    """Analyze all results in the directory"""
    
    M_values = [300, 200, 100, 50, 20, 10]
    
    println("Analysis of THIS Multiple Experiments")
    println("="^80)
    println()
    
    # Store statistics for summary
    all_stats = Dict()
    
    for M in M_values
        M_dir = "$results_dir/M_$M"
        all_trials_file = "$M_dir/all_trials.txt"
        
        if !isfile(all_trials_file)
            println("Warning: $all_trials_file not found, skipping M=$M")
            continue
        end
        
        # Parse results
        results = parse_trial_results(all_trials_file)
        
        # Calculate statistics
        stats = Dict()
        
        println("M = $M")
        println("-"^80)
        println("Order | Mean    | Std     | Variance | Min     | Max     | n")
        println("------+---------+---------+----------+---------+---------+----")
        
        for order in 2:7
            values = results[order]
            n = length(values)
            
            if n > 0
                m = mean(values)
                s = std(values)
                v = var(values)
                min_val = minimum(values)
                max_val = maximum(values)
                
                stats[order] = Dict(
                    "mean" => m,
                    "std" => s,
                    "var" => v,
                    "min" => min_val,
                    "max" => max_val,
                    "n" => n
                )
                
                println(@sprintf("%5d | %.4f | %.4f | %.6f | %.4f | %.4f | %2d",
                    order, m, s, v, min_val, max_val, n))
            else
                println(@sprintf("%5d | No data", order))
            end
        end
        
        all_stats[M] = stats
        println()
        
        # Save statistics to file
        open("$M_dir/statistics.txt", "w") do f
            write(f, "M=$M Statistics\n\n")
            write(f, "Order | Mean    | Std     | Variance | Min     | Max     | n\n")
            write(f, "------+---------+---------+----------+---------+---------+----\n")
            for order in 2:7
                if haskey(stats, order)
                    s = stats[order]
                    write(f, @sprintf("%5d | %.4f | %.4f | %.6f | %.4f | %.4f | %2d\n",
                        order, s["mean"], s["std"], s["var"], s["min"], s["max"], s["n"]))
                end
            end
        end
    end
    
    # Create comprehensive summary
    println("\n" * "="^80)
    println("Summary Across All M Values")
    println("="^80)
    println()
    
    summary_file = "$results_dir/analysis_summary.txt"
    open(summary_file, "w") do f
        write(f, "Comprehensive Analysis Summary\n")
        write(f, "="^80 * "\n\n")
        
        for order in 2:7
            println("Order $order:")
            println("-"^80)
            println("  M   | Mean    | Std     | Variance | Min     | Max")
            println("------+---------+---------+----------+---------+---------")
            
            write(f, "Order $order:\n")
            write(f, "-"^80 * "\n")
            write(f, "  M   | Mean    | Std     | Variance | Min     | Max\n")
            write(f, "------+---------+---------+----------+---------+---------\n")
            
            for M in M_values
                if haskey(all_stats, M) && haskey(all_stats[M], order)
                    s = all_stats[M][order]
                    line = @sprintf("%5d | %.4f | %.4f | %.6f | %.4f | %.4f",
                        M, s["mean"], s["std"], s["var"], s["min"], s["max"])
                    println(line)
                    write(f, line * "\n")
                end
            end
            println()
            write(f, "\n")
        end
    end
    
    println("Analysis complete!")
    println("Summary saved to: $summary_file")
    
    return all_stats
end

# Main execution
if length(ARGS) > 0
    results_dir = ARGS[1]
else
    # Find the most recent results directory
    results_dirs = readdir("results_THIS_multiple", join=true)
    if isempty(results_dirs)
        println("Error: No results directories found in results_THIS_multiple/")
        exit(1)
    end
    results_dir = sort(results_dirs)[end]  # Most recent
    println("Using most recent results directory: $results_dir")
    println()
end

all_stats = analyze_results(results_dir)
