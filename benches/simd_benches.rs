use criterion::{criterion_group, criterion_main, Criterion};
use prefix_scan;


/**
 * Some benchmarks for various simd functions used as building blocks by the prefix scans for speedup.
 */

const LARGE_COUNT: u64 = 10000000;

fn sequential_simd_bench(c: &mut Criterion) {
    c.bench_function("sequential simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        b.iter(move || prefix_scan::baseline::sequential_scan_simd(&mut vec))
    });
}

fn quicksum_simd_bench(c: &mut Criterion) {
    c.bench_function("quicksum simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        b.iter(move || prefix_scan::helper_functions::quicksum_simd(&mut vec))
    });
}

fn sequential_no_simd_bench(c: &mut Criterion) {
    c.bench_function("sequential no simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        b.iter(move || vec = prefix_scan::baseline::sequential_scan_no_simd(std::mem::replace(&mut vec, vec![]), |a, b| a + b).unwrap())
    });
}

fn parallel_simd_quicksum_bench(c: &mut Criterion) {
    c.bench_function("parallel quicksum simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        let mut scanner = prefix_scan::Scanner::new().with_threads(4);
        b.iter(move || scanner.parallel_quicksum_simd(&mut vec))
    });
}

criterion_group!(simd_benches, 
    sequential_simd_bench, 
    quicksum_simd_bench, 
    sequential_no_simd_bench, 
    parallel_simd_quicksum_bench
);

criterion_main!(simd_benches);