use criterion::{criterion_group, criterion_main, Criterion};
use prefix_scan;

const DATA_SIZE: u64 = 5000000;
const NUM_THREADS: usize = 4;
// const CACHE_CHUNK_LENGTH: usize = 250000;
// const SEQUENTIAL_LENGTH: usize = 10000;


fn tune_cache_chunk_length_bench(c: &mut Criterion) {
    let mut scanner = prefix_scan::Scanner::new()
        .with_threads(NUM_THREADS);
    let vec = (0..DATA_SIZE).collect::<Vec<_>>();
    
    let mut group = c.benchmark_group("tune cache chunk length bench");
    for cache_chunk_length in (0..10).map(|i| i * 25000) {
        group.throughput(criterion::Throughput::Bytes(8 * DATA_SIZE));
        group.bench_with_input(criterion::BenchmarkId::from_parameter(cache_chunk_length), &cache_chunk_length, |b, &cache_chunk_length| {
            scanner.set_cache_chunk_length(cache_chunk_length);
            b.iter_batched(
                || vec.clone(),
                |data| scanner.divide_and_conquer_post_scatter_scan(data),
                criterion::BatchSize::LargeInput
            )
        });
    }    
}


criterion_group!(parameter_tuning_benches, tune_cache_chunk_length_bench);
criterion_main!(parameter_tuning_benches);
