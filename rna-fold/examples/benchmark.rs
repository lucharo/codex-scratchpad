//! Benchmark example comparing LinearFold and EternaFold performance
//!
//! Run with: cargo run --example benchmark --release

use rna_fold::{RnaSequence, LinearFold, EternaFold};
use std::time::Instant;

fn benchmark_algorithm<F>(name: &str, sequence: &RnaSequence, fold_fn: F)
where
    F: Fn(&RnaSequence) -> (),
{
    let start = Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        fold_fn(sequence);
    }

    let duration = start.elapsed();
    let avg_time = duration.as_micros() as f64 / iterations as f64;

    println!("{:20} {:>10.2} μs/iter", name, avg_time);
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("RNA Folding Performance Benchmarks");
    println!("{}", "=".repeat(80));

    // Test sequences of different lengths
    let test_cases = vec![
        ("Short (12 nt)", "GGGGAAAACCCC"),
        ("Medium (30 nt)", "GGGGGGAAAAAAAACCCCCCUUUUUUUUUU"),
        ("Long (60 nt)", "GGGGGGGGGGAAAAAAAAAAACCCCCCCCCCUUUUUUUUUUGGGGGGGGGGAAAAAAAAAA"),
    ];

    for (description, seq_str) in test_cases {
        println!("\n{}", description);
        println!("Sequence length: {} nucleotides", seq_str.len());
        println!("{}", "-".repeat(80));

        let sequence = RnaSequence::from_str(seq_str).unwrap();

        // Benchmark LinearFold with different beam sizes
        for beam_size in [10, 50, 100] {
            let folder = LinearFold::new(beam_size);
            let name = format!("LinearFold (b={})", beam_size);
            benchmark_algorithm(&name, &sequence, |seq| {
                let _ = folder.fold(seq);
            });
        }

        // Benchmark EternaFold
        let folder = EternaFold::new();
        benchmark_algorithm("EternaFold", &sequence, |seq| {
            let _ = folder.fold(seq);
        });
    }

    println!("\n{}", "=".repeat(80));
    println!("Notes:");
    println!("- Run with --release for accurate measurements");
    println!("- LinearFold: beam size trades off speed vs accuracy");
    println!("- EternaFold: O(n³) DP, slower but potentially more accurate");
    println!("{}", "=".repeat(80));
}
