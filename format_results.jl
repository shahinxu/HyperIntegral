"""
Format results in compact form:
M=300
Order-2, Order-3, ...
average_1 (std_1), ...
"""

using Statistics
using Printf

function parse_trial_results(filepath)
    results = Dict{Int, Vector{Float64}}()
    for order in 2:7
        results[order] = Float64[]
    end
    
    open(filepath, "r") do f
        for line in eachline(f)
            if occursin("-edges:", line)
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

function format_results(results_dir)
    M_values = [300, 200, 100, 50, 20, 10]
    
    output_file = "$results_dir/formatted_results.txt"
    
    open(output_file, "w") do f
        for M in M_values
            M_dir = "$results_dir/M_$M"
            all_trials_file = "$M_dir/all_trials.txt"
            
            if !isfile(all_trials_file)
                continue
            end
            
            results = parse_trial_results(all_trials_file)
            
            # Write M value
            write(f, "M=$M\n")
            
            # Write header (Order-2, Order-3, ...)
            header = join(["Order-$o" for o in 2:7], ", ")
            write(f, header * "\n")
            
            # Calculate and write averages with std
            stats_strs = String[]
            for order in 2:7
                values = results[order]
                if length(values) > 0
                    avg = mean(values)
                    std_val = std(values)
                    push!(stats_strs, @sprintf("%.4f (%.4f)", avg, std_val))
                else
                    push!(stats_strs, "N/A")
                end
            end
            write(f, join(stats_strs, ", ") * "\n\n")
        end
    end
    
    # Print to console as well
    println("Formatted Results:")
    println("="^80)
    open(output_file, "r") do f
        print(read(f, String))
    end
    
    println("="^80)
    println("Results saved to: $output_file")
end

# Main
if length(ARGS) > 0
    results_dir = ARGS[1]
else
    results_dirs = readdir("results_THIS_multiple", join=true)
    if isempty(results_dirs)
        println("Error: No results directories found")
        exit(1)
    end
    results_dir = sort(results_dirs)[end]
    println("Using: $results_dir\n")
end

format_results(results_dir)
