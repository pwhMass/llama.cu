use llama_cu::{DistKVCache, Message, SampleArgs, Service, Session, SessionId, Terminal, utok};
use std::{collections::BTreeMap, ffi::c_int, iter::zip, path::Path, time::Instant};

fn service(model: impl AsRef<Path>, gpus: &[c_int], use_cuda_grpah: bool) {
    let service = Service::new(model, gpus, use_cuda_grpah);
    let terminal = service.terminal().clone();
    tokio::task::spawn_blocking(move || {
        let mut caches = CacheManager::new(terminal);
        let _ = service;
    });
}

struct CacheManager {
    terminal: Terminal,
    caches: BTreeMap<Instant, (Vec<utok>, DistKVCache)>,
    next_id: usize,
}

impl CacheManager {
    pub fn new(terminal: Terminal) -> Self {
        Self {
            terminal,
            caches: Default::default(),
            next_id: 0,
        }
    }

    pub fn send(&mut self, msgs: &[Message], sample_args: SampleArgs) -> (SessionId, Vec<utok>) {
        let id = SessionId(self.next_id);
        self.next_id += 1;

        let text = self.terminal.render(msgs);
        let tokens = self.terminal.tokenize(&text);

        let best_cache = self
            .caches
            .iter()
            .map(|(key, (history, _))| (*key, common_len(history, &tokens)))
            .max_by_key(|&(_, len)| len);

        let cache = match best_cache {
            Some((key, pos)) => {
                let (_, mut cache) = self.caches.remove(&key).unwrap();
                cache.pos = pos;
                cache
            }
            None => self.terminal.new_cache(),
        };
        let pos = cache.pos;
        self.terminal.start(
            Session {
                id,
                sample_args,
                cache,
            },
            &tokens[pos..],
        );
        (id, tokens)
    }
}

fn common_len<T: Eq>(a: &[T], b: &[T]) -> usize {
    zip(a, b).take_while(|(a, b)| a == b).count()
}
