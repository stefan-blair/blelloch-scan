#![feature(test)]
mod prefix_scan;
mod thread_pool;
mod split_vector;

const LARGE_COUNT: u64 = 128;
const NUM_THREADS: usize = 4;
fn main() {}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::thread_pool;
    use crate::prefix_scan;

    const LARGE_COUNT: u64 = 10000000;
    const NUM_THREADS: usize = 4;

    #[test]
    fn thread_pool_basic_test() {
        let numbers = vec![1, 2, 3, 4];
        let mut pool = thread_pool::ThreadPool::new(4);

        let result: u64 = pool.broadcast(numbers, |(index, _), args: Vec<u64>| {
            args[index] * args[index]
        }).gather().unwrap().iter().sum();

        assert_eq!(result, 1 + 4 + 9 + 16);
    }

    #[test]
    fn large_divide_conquer_test() {
        let count = 12;
        let list = (0..count).collect::<Vec<_>>();
        assert_eq!(prefix_scan::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap(), prefix_scan::Scanner::new().with_threads(4).divide_and_conquer_scan(list).unwrap())
    }

    #[test]
    fn large_divide_conquer_test_2() {
        let count = 12;
        let list = (0..count).collect::<Vec<_>>();
        assert_eq!(prefix_scan::baseline::sequential_scan_no_simd(list.clone(), |a, b| a + b).unwrap(), prefix_scan::Scanner::new().with_threads(4).divide_and_conquer_scan_2(list).unwrap())
    }

    #[test]
    fn simd_sequential_test() {
        let count = 32;
        let mut list = (0..count).collect::<Vec<_>>();
        let baseline = list.clone();
        prefix_scan::baseline::sequential_scan_simd(&mut list[..]).unwrap();
        assert_eq!(prefix_scan::baseline::sequential_scan_no_simd(baseline, |a, b| a + b).unwrap(), list)
    }

    #[test]
    fn quicksum_test() {
        let vec = (0..35).collect::<Vec<_>>();
        assert_eq!(prefix_scan::quicksum_simd(&vec), vec.iter().sum());
    }

    #[test]
    fn parallel_quicksum_test() {
        let vec = (0..35).collect::<Vec<_>>();
        assert_eq!(prefix_scan::Scanner::new().with_threads(4).parallel_quicksum_simd(&vec), vec.iter().sum());        
    }
}
