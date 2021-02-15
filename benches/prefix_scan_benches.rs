use criterion::{criterion_group, criterion_main, Criterion};
use prefix_scan;

const CHUNK_SIZE: u64 = 500000;
const CACHE_CHUNK_LENGTH: usize = 250000;
const NUM_THREADS: usize = 4;
const SEQUENTIAL_LENGTH: usize = 10000;


fn scan_benchmark(c: &mut Criterion, name: &'static str, do_scan: fn(&mut prefix_scan::Scanner, Vec<u64>) -> Result<Vec<u64>, prefix_scan::ScanError>) {
    let mut scanner = prefix_scan::Scanner::new()
        .with_threads(NUM_THREADS)
        .with_cache_chunk_length(CACHE_CHUNK_LENGTH)
        .with_sequential_length(SEQUENTIAL_LENGTH);
    
    let mut group = c.benchmark_group(name);
    for size in [1, 2, 4, 8, 16, 32, 64].iter().map(|i| i * CHUNK_SIZE) {
        group.throughput(criterion::Throughput::Bytes(8 * size));
        group.bench_with_input(criterion::BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vec = (0..size).collect::<Vec<_>>();
            b.iter_batched(
                || vec.clone(),
                |data| do_scan(&mut scanner, data).unwrap(),
                criterion::BatchSize::LargeInput
            )
        });
    }    
}

fn divide_and_conquer_post_scatter_bench(c: &mut Criterion) {
    scan_benchmark(c, "divide conquer post scatter bench", |scanner, data| scanner.divide_and_conquer_post_scatter_scan(data))
}

fn divide_and_conquer_pre_scatter_bench(c: &mut Criterion) {
    scan_benchmark(c, "divide conquer pre scatter bench", |scanner, data| scanner.divide_and_conquer_pre_scatter_scan(data))
}

fn hillis_steel_bench(c: &mut Criterion) {
    scan_benchmark(c, "hillis steel bench", |scanner, data| scanner.hillis_steel_scan(data))
}

fn blelloch_bench(c: &mut Criterion) {
    scan_benchmark(c, "blelloch bench", |scanner, data| scanner.blelloch_scan(data, |a, b| a + b))
}

fn sequential_baseline_bench(c: &mut Criterion) {
    scan_benchmark(c, "sequential baseline bench", |_, mut data| Ok(prefix_scan::baseline::sequential_scan_simd(&mut data)).map(|_| data))
}

criterion_group!(prefix_scan_benches, 
    divide_and_conquer_post_scatter_bench, 
    divide_and_conquer_pre_scatter_bench,
    blelloch_bench,
    hillis_steel_bench,
    sequential_baseline_bench
);

criterion_main!(prefix_scan_benches);
