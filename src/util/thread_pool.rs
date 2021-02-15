use std::thread;
use std::sync::mpsc;
use std::ops::{Index, IndexMut};


/**
 * A thread id is a tuple, (thread's individual index, total number of threads)
 */
type ThreadId = (usize, usize);
/**
 * A specific function signature that takes in a thread id and argument, and produces some return value
 */
type ThreadFunction<S, R> = fn(ThreadId, S) -> R;
/**
 * Shorthand for a channel that returns a tuple, (thread id, result)
 */
type ThreadSendResultChannel<R> = mpsc::Sender<(ThreadId, R)>;
type ThreadReceiveResultChannel<R> = mpsc::Receiver<(ThreadId, R)>;

trait Callable {
    fn call(self: Box<Self>);
}

pub struct ThreadWork<S, R> {
    argument: Box<S>,
    function: ThreadFunction<S, R>,
    send_channel: ThreadSendResultChannel<R>,
    thread_id: ThreadId,
}

impl<S, R> ThreadWork<S, R> {
    fn new(argument: S, function: ThreadFunction<S, R>, send_channel: ThreadSendResultChannel<R>, thread_id: ThreadId) -> Self {
        Self { argument: Box::new(argument), function, send_channel, thread_id }
    }
}

impl<S: 'static + Send, R> Callable for ThreadWork<S, R> {
    fn call(self: Box<Self>) {
        self.send_channel.send((self.thread_id, (self.function)(self.thread_id, *self.argument))).unwrap();
    }
}

pub struct RemoteThread {
    _handle: thread::JoinHandle<()>,
    send_channel: mpsc::Sender<Box<dyn Callable + Send>>,
}

impl RemoteThread {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Box<dyn Callable + Send>>();
        let handle = thread::spawn(move || {
            for msg in rx {
                msg.call();
            }
        });
        
        Self { _handle: handle, send_channel: tx }
    }

    fn send<S: 'static + Send, R: 'static + Send>(&mut self, function: ThreadFunction<S, R>, msg: S, result_channel: ThreadSendResultChannel<R>, thread_id: ThreadId) {
        let work = Box::new(ThreadWork::new(msg, function, result_channel, thread_id));
        self.send_channel.send(work).unwrap();
    }
}

pub enum Thread {
    Remote(RemoteThread),
    Local,
}

impl Thread {
    fn remote() -> Self {
        Self::Remote(RemoteThread::new())
    }

    fn local() -> Self {
        Self::Local
    }

    pub fn send<S: 'static + Send, R: 'static + Send>(&mut self, function: ThreadFunction<S, R>, msg: S, result_channel: ThreadSendResultChannel<R>, thread_id: ThreadId) {
        match self {
            Self::Remote(r) => r.send(function, msg, result_channel, thread_id),
            Self::Local => result_channel.send((thread_id, function(thread_id, msg))).unwrap()
        }
    }
}

pub struct MassReceiver<R> {
    receiver: ThreadReceiveResultChannel<R>,
    expected_msg_count: usize,
}

impl<R: 'static + Send> MassReceiver<R> {
    fn new(receiver: ThreadReceiveResultChannel<R>, expected_msg_count: usize) -> Self {
        Self { receiver, expected_msg_count }
    }

    pub fn gather(self) -> Result::<Vec<R>, mpsc::RecvError> {
        let mut results = (0..self.expected_msg_count).map(|_| None).collect::<Vec<_>>();
        for _ in 0..self.expected_msg_count {
            let ((index, _), msg) = self.receiver.recv()?;
            match results[index] {
                None => results[index] = Some(msg),
                Some(_) => return Err(mpsc::RecvError)
            }
        }

        Ok(results.into_iter().map(|x| x.unwrap()).collect())
    }
}

pub struct ThreadPool {
    threads: Vec<Thread>
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let mut threads = (0..(num_threads - 1)).map(|_| Thread::remote()).collect::<Vec<_>>();
        threads.push(Thread::local());

        Self { threads }
    }

    pub fn sendall<S: 'static + Send, R: 'static + Send>(&mut self, msgs: Vec<S>, function: ThreadFunction<S, R>) -> MassReceiver<R> {
        let (tx, rx) = mpsc::channel();
        let msg_count = msgs.len();
        let num_threads = self.threads.len();

        for (i, msg) in msgs.into_iter().enumerate() {
            self.threads[i].send(function, msg, mpsc::Sender::clone(&tx), (i, num_threads));
        }

        MassReceiver::new(rx, msg_count)
    }

    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }

    pub fn broadcast<S: 'static + Send + Clone, R: 'static + Send>(&mut self, msg: S, function: ThreadFunction<S, R>) -> MassReceiver<R>{
        let (tx, rx) = mpsc::channel();
        let num_threads = self.threads.len();

        for (i, thread) in self.threads.iter_mut().enumerate() {
            thread.send(function, msg.clone(), mpsc::Sender::clone(&tx), (i, num_threads));
        }

        MassReceiver::new(rx, self.threads.len())
    }
}

impl Index<usize> for ThreadPool {
    type Output = Thread;

    fn index(&self, index: usize) -> &Self::Output {
        &self.threads[index]
    }
}

impl IndexMut<usize> for ThreadPool {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.threads[index]
    }
}

#[cfg(test)]
mod test {
    use crate::util::thread_pool;

    #[test]
    fn thread_pool_basic_test() {
        let numbers = vec![1, 2, 3, 4];
        let mut pool = thread_pool::ThreadPool::new(4);

        let result: u64 = pool.broadcast(numbers, |(index, _), args: Vec<u64>| {
            args[index] * args[index]
        }).gather().unwrap().iter().sum();

        assert_eq!(result, 1 + 4 + 9 + 16);
    }
}