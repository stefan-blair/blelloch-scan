use criterion::{criterion_group, criterion_main, Criterion};
use prefix_scan;

/**
 * Configuration of the scanner across all benchmarks.
 */
const DATA_SIZE: u64 = 5000000;
const CACHE_CHUNK_LENGTH: usize = 250000;
const NUM_THREADS: usize = 8;
const SEQUENTIAL_LENGTH: usize = 10000;


/**
 * Iterates over each of the different scan algorithms and runs them on the same dataset in the same benchmark group for easy comparison.
 */
fn prefix_scans_bench(c: &mut Criterion) {
    // initialize the scanner
    let mut scanner = prefix_scan::Scanner::new()
        .with_threads(NUM_THREADS)
        .with_cache_chunk_length(CACHE_CHUNK_LENGTH)
        .with_sequential_length(SEQUENTIAL_LENGTH);

    // vector of pairs of (algorithm name, algorithm caller)
    let scan_algorithms: Vec<(&str, fn(&mut prefix_scan::Scanner, Vec<u64>) -> Result<Vec<u64>, prefix_scan::ScanError>)> = vec![
        ("divide conquer post scatter bench", |scanner, data| scanner.divide_and_conquer_scan(data)),
        ("hillis steel bench", |scanner, data| scanner.hillis_steel_scan(data)),
        ("blelloch bench", |scanner, data| scanner.blelloch_scan(data)),
        ("sequential baseline bench", |_, mut data| Ok(prefix_scan::baseline::sequential_scan_simd(&mut data)).map(|_| data))
    ];
    
    let mut group = c.benchmark_group("prefix scan benches");
    for (name, caller) in scan_algorithms {
        group.throughput(criterion::Throughput::Bytes(8 * DATA_SIZE));
        group.bench_with_input(criterion::BenchmarkId::from_parameter(name), &caller, |b, &caller| {
            let vec = (0..DATA_SIZE).collect::<Vec<u64>>();
            b.iter_batched(
                || vec.clone(),
                |data| caller(&mut scanner, data).unwrap(),
                criterion::BatchSize::LargeInput
            )
        });
    }    
}

criterion_group!(prefix_scans, prefix_scans_bench);

criterion_main!(prefix_scans);
