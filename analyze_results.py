from utils.plotting import plot_algorithm_comparison, create_results_table

if __name__ == "__main__":
    results_dir = "experiments/results"

    print("Analyzing experiment results...\n")

    # Create comparison plot
    plot_algorithm_comparison(results_dir)

    # Create results table
    create_results_table(results_dir)

    print("\nAnalysis complete!")
