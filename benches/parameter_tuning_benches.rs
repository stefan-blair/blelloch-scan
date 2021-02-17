use criterion::{criterion_group, criterion_main, Criterion};
use prefix_scan;

const DATA_SIZE: u64 = 5000000;
const NUM_THREADS: usize = 4;
const SEQUENTIAL_LENGTH: usize = 10000;

/**
 * Benchmarks tuning the individual parameters of the scanner to find which values are most efficient.
 */

fn tune_cache_chunk_length_bench(c: &mut Criterion) {
    let mut scanner = prefix_scan::Scanner::new()
        .with_threads(NUM_THREADS);
    let vec = (0..DATA_SIZE).collect::<Vec<u64>>();
    
    let mut group = c.benchmark_group("tune cache chunk length bench");
    for cache_chunk_length in (1..10).map(|i| i * 25000) {
        group.throughput(criterion::Throughput::Bytes(8 * DATA_SIZE));
        group.bench_with_input(criterion::BenchmarkId::from_parameter(cache_chunk_length), &cache_chunk_length, |b, &cache_chunk_length| {
            scanner.set_cache_chunk_length(cache_chunk_length);
            b.iter_batched(
                || vec.clone(),
                |data| scanner.divide_and_conquer_scan(data),
                criterion::BatchSize::LargeInput
            )
        });
    }    
}

fn tune_sequential_length_bench(c: &mut Criterion) {
    let mut scanner = prefix_scan::Scanner::new()
        .with_threads(NUM_THREADS);
    let vec = (0..DATA_SIZE).collect::<Vec<u64>>();
    
    let mut group = c.benchmark_group("tune sequential length bench");
    for sequential_length in (0..10).map(|i| i * 2000) {
        group.throughput(criterion::Throughput::Bytes(8 * DATA_SIZE));
        group.bench_with_input(criterion::BenchmarkId::from_parameter(sequential_length), &sequential_length, |b, &sequential_length| {
            scanner.set_sequential_length(sequential_length);
            b.iter_batched(
                || vec.clone(),
                |data| scanner.blelloch_scan(data),
                criterion::BatchSize::LargeInput
            )
        });
    }    
}

fn tune_num_threads_bench(c: &mut Criterion) {
    let vec = (0..DATA_SIZE).collect::<Vec<u64>>();
    
    let mut group = c.benchmark_group("tune num threads bench");
    for num_threads in 1..10 {
        group.throughput(criterion::Throughput::Bytes(8 * DATA_SIZE));
        let mut scanner = prefix_scan::Scanner::new()
            .with_threads(num_threads)
            .with_sequential_length(SEQUENTIAL_LENGTH);
        group.bench_with_input(criterion::BenchmarkId::from_parameter(num_threads), &num_threads, |b, _| {
            b.iter_batched(
                || vec.clone(),
                |data| scanner.blelloch_scan(data),
                criterion::BatchSize::LargeInput
            )
        });
    }    
}

criterion_group!(parameter_tuning_benches, 
    tune_cache_chunk_length_bench, 
    tune_sequential_length_bench,
    tune_num_threads_bench
);
criterion_main!(parameter_tuning_benches);
