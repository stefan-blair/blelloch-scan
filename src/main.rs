#![feature(test)]
mod segmented_scan;
mod thread_pool;
mod split_vector;

const LARGE_COUNT: u64 = 128;
const NUM_THREADS: usize = 4;
fn main() {

    for i in 128..256 {
        let vec = (1..i+1).collect::<Vec<_>>();

        let blelloch = segmented_scan::blelloch_scan(NUM_THREADS, &vec).unwrap();
        let sequential = segmented_scan::sequential_scan(&vec![vec]).unwrap();
        assert_eq!(&blelloch[1..], sequential[0].split_last().unwrap().1);
    }

    // let segmented_list = vec![(0..LARGE_COUNT).collect::<Vec<_>>()];
    // segmented_scan::divide_and_conquer_scan(NUM_THREADS, &segmented_list).unwrap();

    // let segmented_list = vec![(0..LARGE_COUNT).collect::<Vec<_>>()];
    // segmented_scan::sequential_scan(&segmented_list);
}

#[cfg(test)]
mod tests {
    extern crate test;

    use test::Bencher;

    use crate::thread_pool;
    use crate::segmented_scan;

    const LARGE_COUNT: u64 = 100000000;
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
    fn large_hillis_steele_test() {
        let count = 12;
        let segmented_list = vec![(0..count).collect::<Vec<_>>()];
        println!("segmented_list: {:?}", segmented_list);
        assert_eq!(segmented_scan::sequential_scan(&segmented_list), segmented_scan::hillis_steel_scan(NUM_THREADS, &segmented_list))
    }

    #[test]
    fn large_divide_conquer_test() {
        let count = 12;
        let segmented_list = vec![(0..count).collect::<Vec<_>>()];
        println!("segmented_list: {:?}", segmented_list);
        assert_eq!(segmented_scan::sequential_scan(&segmented_list), segmented_scan::divide_and_conquer_scan(NUM_THREADS, &segmented_list))
    }

    #[ignore]
    #[bench]
    fn large_hillis_steele_scan(b: &mut Bencher) {
        let segmented_list = vec![(0..LARGE_COUNT).collect::<Vec<_>>()];
        b.iter(move|| {
            segmented_scan::hillis_steel_scan(NUM_THREADS, &segmented_list)
        })
    }

    #[bench]
    fn large_divide_conquer_scan(b: &mut Bencher) {
        let segmented_list = vec![(0..LARGE_COUNT).collect::<Vec<_>>()];
        b.iter(move|| {
            segmented_scan::divide_and_conquer_scan(NUM_THREADS, &segmented_list)
        })
    }

    #[bench]
    fn large_sequential_scan(b: &mut Bencher) {
        let segmented_list = vec![(0..LARGE_COUNT).collect::<Vec<_>>()];
        b.iter(move|| {
            segmented_scan::sequential_scan(&segmented_list)
        })
    }
}
