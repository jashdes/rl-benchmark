from utils.multiseed_plotting import (
    plot_multiseed_comparison,
    plot_seed_distribution,
    create_multiseed_summary,
)


if __name__ == "__main__":
    results_dir = "experiments/results"

    print("Analyzing multi-seed experiment results...\n")

    # Create comparison with error bars
    plot_multiseed_comparison(results_dir)

    # Create box plot showing distributions
    plot_seed_distribution(results_dir)

    # Create summary table
    create_multiseed_summary(results_dir)

    print("\nMulti-seed analysis complete!")
