//! Basic usage example for the RNA folding library
//!
//! Run with: cargo run --example basic_usage

use rna_fold::{RnaSequence, LinearFold, EternaFold};

fn main() {
    println!("{}", "=".repeat(80));
    println!("RNA Secondary Structure Prediction Examples");
    println!("{}", "=".repeat(80));

    // Example sequences
    let sequences = vec![
        ("Simple hairpin", "GGGGAAAACCCC"),
        ("Small stem-loop", "CGCGAAAAAAGCGCG"),
        ("Complex structure", "GGGGGGAAAACCCCCC"),
    ];

    for (name, seq_str) in sequences {
        println!("\n{}", name);
        println!("Sequence: {}", seq_str);
        println!("Length: {} nucleotides", seq_str.len());

        // Parse sequence
        let sequence = match RnaSequence::from_str(seq_str) {
            Ok(seq) => seq,
            Err(e) => {
                eprintln!("Error parsing sequence: {}", e);
                continue;
            }
        };

        // LinearFold prediction
        println!("\n--- LinearFold (beam search, O(n) time) ---");
        let linearfold = LinearFold::new(100); // beam size
        let structure_lf = linearfold.fold(&sequence);
        println!("{}", structure_lf);

        // EternaFold prediction
        println!("\n--- EternaFold (ML-based) ---");
        let eternafold = EternaFold::new();
        let structure_ef = eternafold.fold(&sequence);
        println!("{}", structure_ef);

        // Partition function
        let z = eternafold.partition_function(&sequence);
        println!("Partition function: {:.2e}", z);

        println!("\n{}", "-".repeat(80));
    }

    // Demonstrate error handling
    println!("\n{}", "=".repeat(80));
    println!("Error Handling Example");
    println!("{}", "=".repeat(80));

    let invalid_seq = "GGGXAAACCC";  // X is invalid
    match RnaSequence::from_str(invalid_seq) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("✓ Caught expected error: {}", e),
    }

    println!("\nDone!");
}
