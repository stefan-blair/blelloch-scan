
#[derive(Clone, Debug)]
pub struct Range<T> {
    start: usize,
    end: usize,
    value: T,
    index: usize,
}

impl<T> Range<T> {
    fn new(start: usize, end: usize, value: T, index: usize) -> Self {
        Self { start, end, index, value }
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }
}

#[derive(Clone, Debug)]
pub struct RangedVector<T> {
    ranges: Vec<Range<T>>
}

impl<T> RangedVector<T> {
    pub fn new(points: Vec<usize>, values: Vec<T>) -> Self {
        Self { 
            ranges: values.into_iter().enumerate().map(|(i, value)| Range::new(points[i], points[i + 1], value, i)).collect::<Vec<_>>()
        }
    }

    pub fn get(&self, point: usize) -> Option<&Range<T>> {
        for range in self.ranges.iter() {
            if point < range.end {
                return Some(range)
            }
        }

        return None
    }

    pub fn get_range(&self, index: usize) -> Option<&Range<T>> {
        if index < self.ranges.len() {
            Some(&self.ranges[index])
        } else {
            None
        }
    }

    pub fn next_range(&self, range: &Range<T>) -> Option<&Range<T>> {
        if range.index < self.ranges.len() - 1 {
            Some(&self.ranges[range.index + 1])
        } else {
            None
        }
    }
}