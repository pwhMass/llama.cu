mod error;
mod openai;
mod response;

use crate::BaseArgs;
use error::Error;
use http_body_util::{BodyExt, combinators::BoxBody};
use hyper::{
    Method, Request, Response,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use llama_cu::{OwnedMessage, Received, Service, Session, SessionId, Terminal, TextBuf};
use log::{info, warn};
use openai::{Completions, CompletionsChoice, CompletionsResponse, V1_COMPLETIONS_OBJECT};
use openai_struct::CreateCompletionResponse;
use openai_struct::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use openai_struct::{CreateCompletionRequest, ModelIdsShared};

use response::{error, text_stream};
use std::{
    collections::BTreeMap,
    ffi::c_int,
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    net::TcpListener,
    sync::mpsc::{self, UnboundedSender},
};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Args)]
pub struct ServiceArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    port: u16,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self { base, port } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(start_infer_service(
                base.model,
                port,
                gpus,
                max_steps,
                !base.no_cuda_graph,
            ))
            .unwrap()
    }
}

fn get_session_id() -> SessionId {
    static SESSION_ID: AtomicUsize = AtomicUsize::new(0);
    SessionId(SESSION_ID.fetch_add(1, SeqCst))
}

async fn start_infer_service(
    model: PathBuf,
    port: u16,
    gpus: Box<[c_int]>,
    max_steps: usize,
    use_cuda_graph: bool,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let service = Service::new(model, &gpus, use_cuda_graph);
    let sessions: BTreeMap<SessionId, SessionInfo> = BTreeMap::new();

    let service_manager = Arc::new(ServiceManager {
        terminal: service.terminal().clone(),
        max_steps,
        sessions: Mutex::new(sessions),
    });

    // 将所有recv逻辑移到这个handle中
    let service_manager_for_recv = service_manager.clone();
    let _send_handle = tokio::spawn(async move {
        loop {
            let received = service.recv();

            let Received {
                sessions: completed_sessions,
                outputs,
            } = received;

            // 处理输出
            for (session_id, tokens) in outputs {
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                if let Some(session_info) = sessions_guard.get_mut(&session_id) {
                    let text = service_manager_for_recv
                        .terminal
                        .decode(&tokens, &mut session_info.buf);
                    info!("发送文本: {}", text);
                    if session_info.sender.send(text.to_string()).is_err() {
                        info!("发送失败，可能是接收端已关闭");
                        // 发送失败，可能是接收端已关闭
                        break;
                    }
                }
            }

            // 清理已完成的会话
            if !completed_sessions.is_empty() {
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                for (session, _) in completed_sessions {
                    sessions_guard.remove(&session.id);
                }
            }
        }
    });

    let app = App(service_manager);

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        let (stream, x) = listener.accept().await?;
        info!("listen from {x}");
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}")
            }
        });
    }
}

struct SessionInfo {
    sender: UnboundedSender<String>,
    buf: TextBuf,
    /// 剩余步数
    remaining_steps: usize,
}

struct ServiceManager {
    terminal: Terminal,
    max_steps: usize,
    sessions: Mutex<BTreeMap<SessionId, SessionInfo>>,
}

#[derive(Clone)]
struct App(Arc<ServiceManager>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let service_manager = self.0.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, openai::V1_COMPLETIONS) => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body);
                Ok(match req {
                    Ok(completions) => complete(completions, service_manager),
                    Err(e) => error(Error::WrongJson(e)),
                })
            }),
            // Return 404 Not Found for other routes.
            (method, uri) => {
                let msg = Error::not_found(method, uri);
                Box::pin(async move { Ok(error(msg)) })
            }
        }
    }
}

fn complete(
    completions: CreateChatCompletionRequest,
    service_manager: Arc<ServiceManager>,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    let CreateChatCompletionRequest {
        // TODO 目前并未用到model，并且缺少对其他参数的支持
        model: _model,
        messages,
        max_tokens,
        ..
    } = completions;
    let (sender, receiver) = mpsc::unbounded_channel();

    info!("completions: ");
    // tokio::task::spawn_blocking(move || {

    // static ID: AtomicUsize = AtomicUsize::new(0);
    // let id = format!("InfiniLM-{:#x}", ID.fetch_add(1, SeqCst));
    // let created = SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_secs() as _;

    // for _ in 0..max_steps {
    //     let service_guard = service_data.lock().unwrap();
    //     let service = &*service_guard;
    //     let Received { sessions, outputs } = service.recv();
    //     drop(service_guard);

    //     for (_, (_, piece)) in outputs {
    //         let text = unsafe { std::str::from_utf8_unchecked(&piece) };
    //         let response = CompletionsResponse {
    //             id: id.clone(),
    //             choices: vec![CompletionsChoice {
    //                 index: 0,
    //                 text: text.to_string(),
    //             }],
    //             created,
    //             model: model.clone(),
    //             object: V1_COMPLETIONS_OBJECT.into(),
    //         };
    //         let msg = serde_json::to_string(&response).unwrap();
    //         if sender.send(msg).is_err() {
    //             break;
    //         }
    //     }

    //     if !sessions.is_empty() {
    //         break;
    //     }
    // }
    // });

    let session_id = {
        let session_id = get_session_id();
        let mut sessions_guard = service_manager.sessions.lock().unwrap();
        sessions_guard.insert(
            session_id,
            SessionInfo {
                sender,
                buf: TextBuf::new(),
                remaining_steps: max_tokens
                    .map_or(service_manager.max_steps, |max_tokens| max_tokens as usize),
            },
        );
        session_id
    };

    {
        let session = Session {
            id: session_id,
            sample_args: Default::default(),
            cache: service_manager.terminal.new_cache(),
        };
        let messages = messages
            .into_iter()
            .map(|m| match m {
                ChatCompletionRequestMessage::User(content) => OwnedMessage {
                    role: "user".into(),
                    content: content.content.to_string(),
                },
                ChatCompletionRequestMessage::System(content) => OwnedMessage {
                    role: "system".into(),
                    content: content.content.to_string(),
                },
                _ => {
                    panic!("unsupported message type {:?}", m);
                }
            })
            .collect::<Vec<_>>();
        service_manager.terminal.start_chat(session, &messages);
    }

    println!("text_stream");
    text_stream(UnboundedReceiverStream::new(receiver))
}

#[test]
fn test_post() {
    use crate::macros::print_now;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use tokio_stream::StreamExt;

    use crate::logger;

    logger::init();

    let Some(path) = std::env::var_os("TEST_MODEL") else {
        println!("TEST_MODE not set");
        return;
    };
    const PORT: u16 = 27001;

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let client = reqwest::Client::new();

            let _handle = tokio::spawn(start_infer_service(
                path.into(),
                PORT,
                [0].into(),
                256,
                false,
            ));

            let rt = tokio::runtime::Handle::current();
            let thread_count = rt.metrics().num_workers();
            println!("当前Runtime中的工作线程数量: {}", thread_count);

            let mut headers: HeaderMap = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            let req_body = serde_json::to_string(&CreateChatCompletionRequest {
                model: ModelIdsShared {},
                messages: vec![ChatCompletionRequestMessage::User(
                    openai_struct::ChatCompletionRequestUserMessage {
                        content: serde_json::Value::String("Once upon a time,".to_string()),
                        name: None,
                    },
                )],
                metadata: None,
                service_tier: None,
                audio: None,
                function_call: None,
                functions: None,
                max_completion_tokens: None,
                max_tokens: Some(256),
                modalities: None,
                n: None,
                parallel_tool_calls: None,
                prediction: None,
                reasoning_effort: None,
                response_format: None,
                store: None,
                tool_choice: None,
                tools: None,
                top_logprobs: None,
                web_search_options: None,
                frequency_penalty: None,
                logit_bias: None,
                logprobs: None,
                presence_penalty: None,
                seed: None,
                stop: None,
                stream: None,
                stream_options: None,
                temperature: None,
                top_p: None,
                user: None,
            })
            .unwrap();

            let sss = ChatCompletionRequestMessage::User(
                openai_struct::ChatCompletionRequestUserMessage {
                    content: serde_json::Value::String("Once upon a time,".to_string()),
                    name: None,
                },
            );

            let sss = serde_json::to_string(&sss).unwrap();

            println!("req: {sss:?}");

            let req = client
                .post(format!("http://localhost:{PORT}{}", openai::V1_COMPLETIONS))
                .headers(headers)
                .body(req_body)
                .timeout(Duration::from_secs(10));

            tokio::time::sleep(Duration::from_secs(30)).await;
            println!("send req");
            let res = req.send().await.unwrap();
            // println!("res: {res:?}");

            // for i in 0..10 {
            //     if res.status().is_success() {
            // }
            if res.status().is_success() {
                let mut stream = res.bytes_stream();
                while let Some(item) = stream.next().await {
                    let text = item.unwrap();
                    let text = std::str::from_utf8(&text).unwrap();
                    print_now!("{text}")
                }
            } else {
                println!("{res:?}");
                let text = res.bytes().await.unwrap();
                let text = std::str::from_utf8(&text).unwrap();
                println!("body: {text}")
            }
        })
}
