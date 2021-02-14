use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prefix_scan;


const LARGE_COUNT: u64 = 10000000;

fn sequential_simd_bench(c: &mut Criterion) {
    c.bench_function("sequential simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        b.iter(move || prefix_scan::segmented_scan::sequential_scan(&mut vec))
    });
}

fn quicksum_simd_bench(c: &mut Criterion) {
    c.bench_function("quicksum simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        b.iter(move || prefix_scan::segmented_scan::quicksum_simd(&mut vec))
    });
}

fn sequential_no_simd_bench(c: &mut Criterion) {
    c.bench_function("sequential no simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        b.iter(move || vec = prefix_scan::segmented_scan::sequential_scan_no_simd(std::mem::replace(&mut vec, vec![]), |a, b| a + b).unwrap())
    });
}

fn parallel_simd_quicksum_bench(c: &mut Criterion) {
    c.bench_function("parallel quicksum simd", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        let mut scanner = prefix_scan::segmented_scan::Scanner::new().with_threads(4);
        b.iter(move || scanner.parallel_quicksum_simd(&mut vec))
    });
}

fn large_divide_conquer_bench(c: &mut Criterion) {
    c.bench_function("large divide conquer bench", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        let mut scanner = prefix_scan::segmented_scan::Scanner::new()
            .with_threads(4)
            .with_cache_chunk_length(250000);
        b.iter(move || vec = scanner.divide_and_conquer_scan(std::mem::replace(&mut vec, vec![])).unwrap())
    });
}

fn large_divide_conquer_bench_2(c: &mut Criterion) {
    c.bench_function("large divide conquer bench", |b| {
        let mut vec = (0..LARGE_COUNT).collect::<Vec<_>>();
        let mut scanner = prefix_scan::segmented_scan::Scanner::new()
            .with_threads(4)
            .with_cache_chunk_length(250000);
        b.iter(move || vec = scanner.divide_and_conquer_scan_2(std::mem::replace(&mut vec, vec![])).unwrap())
    });
}

// #[bench]
// fn sequential_baseline(b: &mut Bencher) {
//     b.iter(move || {
//         let vec = (0..LARGE_COUNT).collect::<Vec<u64>>();
//         assert!(segmented_scan::sequential_scan_no_simd(vec, |a, b| a + b).is_ok());
//     });
// }


criterion_group!(simd_benches, sequential_simd_bench, quicksum_simd_bench, sequential_no_simd_bench, parallel_simd_quicksum_bench, large_divide_conquer_bench, large_divide_conquer_bench_2);
criterion_main!(simd_benches);



// #[ignore]
// #[bench]
// fn large_hillis_steele_scan(b: &mut Bencher) {
//     b.iter(move|| {
//         let segmented_list = vec![(0..LARGE_COUNT).collect::<Vec<_>>()];
//         segmented_scan::Scanner::new()
//             .with_threads(NUM_THREADS)
//             .hillis_steel_scan(&segmented_list)
//     })
// }

// #[ignore]
// #[bench]
// fn large_divide_conquer_scan(b: &mut Bencher) {
//     b.iter(move|| {
//         let segmented_list = (0..LARGE_COUNT).collect::<Vec<_>>();
//         segmented_scan::Scanner::new()
//             .with_threads(NUM_THREADS)
//             .with_cache_chunk_length(250000)
//             .divide_and_conquer_scan(segmented_list)
//     })
// }

// #[ignore]
// #[bench]
// fn large_divide_conquer_scan_no_simd(b: &mut Bencher) {
//     b.iter(move|| {
//         let segmented_list = (0..LARGE_COUNT).collect::<Vec<_>>();
//         segmented_scan::Scanner::new()
//             .with_threads(NUM_THREADS)
//             .with_cache_chunk_length(250000)
//             .without_simd()
//             .divide_and_conquer_scan(segmented_list)
//     })
// }

// #[ignore]
// #[bench]
// fn blelloch_scan(b: &mut Bencher) {
//     b.iter(|| {
//         // the 10000 sequential scan value clocked the highest
//         let vec = (0..LARGE_COUNT).collect::<Vec<_>>();
//         assert!(segmented_scan::Scanner::new()
//             .with_threads(NUM_THREADS)
//             .with_sequential_length(10000)
//             .blelloch_scan(vec, |a, b| a + b).is_ok());
//     });
// }
