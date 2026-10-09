use forge_runtime::chat::Message;
use forge_runtime::ffi::abi::RuntimeHandle;
use std::io::{self, BufRead, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::time::Duration;

const HELP: &str =
    "forge-cli -m MODEL [-c CONTEXT] [-n TOKENS] [--device cpu|cuda] [--dtype fp32|bf16]\n\
Options: -m/--model, -c/--ctx-size (1024), -n/--predict (128), --device (cpu), --dtype (fp32)\n\
Commands: /clear, /exit, /help. Ctrl+C stops generation; at the prompt it exits.";

struct Options {
    model: String,
    context: usize,
    predict: usize,
    device: i32,
}

fn options(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, String> {
    let mut model = "models/Qwen2.5-0.5B-Instruct".to_owned();
    let mut context = 1024;
    let mut predict = 128;
    let mut device = "cpu".to_owned();
    let mut dtype = "fp32".to_owned();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "-h" || arg == "--help" {
            return Ok(None);
        }
        let (flag, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(k, v)| (k, Some(v)));
        if !matches!(
            flag,
            "-m" | "--model" | "-c" | "--ctx-size" | "-n" | "--predict" | "--device" | "--dtype"
        ) {
            return Err(format!("unknown option {flag}; use --help"));
        }
        let value = match inline {
            Some(v) => v.to_owned(),
            None => args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?,
        };
        match flag {
            "-m" | "--model" => model = value,
            "--device" => device = value,
            "--dtype" => dtype = value,
            _ => {
                let n = value
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| format!("{flag} requires a positive integer"))?;
                if matches!(flag, "-c" | "--ctx-size") {
                    context = n;
                } else {
                    predict = n;
                }
            }
        }
    }
    let device = match (device.as_str(), dtype.as_str()) {
        ("cpu", "fp32") => 0,
        ("cuda", "fp32") => 1,
        ("cuda", "bf16") => 2,
        _ => {
            return Err(format!(
                "unsupported device/dtype {device}/{dtype}; use cpu/fp32 or cuda/fp32|bf16"
            ))
        }
    };
    Ok(Some(Options {
        model,
        context,
        predict,
        device,
    }))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(opt) = options(std::env::args().skip(1))? else {
        println!("{HELP}");
        return Ok(());
    };
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal = interrupted.clone();
    ctrlc::set_handler(move || {
        signal.store(true, Ordering::SeqCst);
    })?;
    eprintln!("Loading {} ...", opt.model);
    let mut runtime = RuntimeHandle::load(&opt.model, opt.context, opt.device)?;
    let template = runtime.chat_template()?;
    eprintln!(
        "Ready (context {}). /help for commands.",
        runtime.context_size()
    );
    // A blocking stdin read must not prevent interrupt handling at the prompt.
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let mut history = Vec::new();
    let mut output = io::stdout().lock();
    loop {
        write!(output, "> ")?;
        output.flush()?;
        let input = loop {
            if interrupted.swap(false, Ordering::SeqCst) {
                writeln!(output)?;
                return Ok(());
            }
            match receiver.recv_timeout(Duration::from_millis(50)) {
                Ok(line) => break line?,
                Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        };
        match input.trim() {
            "" => continue,
            "/exit" => return Ok(()),
            "/clear" => {
                history.clear();
                continue;
            }
            "/help" => {
                writeln!(output, "{HELP}")?;
                continue;
            }
            command if command.starts_with('/') => {
                eprintln!("Unknown command; use /help.");
                continue;
            }
            _ => {}
        }
        let mut turns = history.clone();
        turns.push(Message {
            role: "user".into(),
            content: input,
        });
        let prompt = template.render(&turns)?;
        let count = runtime.count_tokens(&prompt)?;
        if count >= runtime.context_size() {
            eprintln!("Context full: {count} formatted tokens, limit {}. Use /clear or a larger --ctx-size.", runtime.context_size());
            continue;
        }
        let limit = opt.predict.min(runtime.context_size() - count);
        let id = match runtime.chat_begin(&prompt, limit) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Generation: {e}");
                continue;
            }
        };
        let mut text = String::new();
        let mut stopped = false;
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            loop {
                if interrupted.swap(false, Ordering::SeqCst) {
                    stopped = true;
                    break;
                }
                let events = runtime.chat_step()?;
                let mut done = false;
                for event in events {
                    write!(output, "{}", event.piece)?;
                    output.flush()?;
                    text.push_str(&event.piece);
                    done |= event.done;
                }
                if done {
                    break;
                }
            }
            Ok(())
        })();
        // Finished slots are already free; cancellation also covers errors/interrupts.
        let _ = runtime.chat_cancel(id);
        result?;
        writeln!(output)?;
        output.flush()?;
        if stopped {
            eprintln!("Generation stopped.");
        }
        turns.push(Message {
            role: "assistant".into(),
            content: text,
        });
        history = turns;
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("forge-cli: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(args: &[&str]) -> Result<Option<Options>, String> {
        options(args.iter().map(|a| a.to_string()))
    }
    #[test]
    fn flags_and_invalid_configurations() {
        let o = parse(&["-m", "model", "--ctx-size=32", "-n", "7"])
            .unwrap()
            .unwrap();
        assert_eq!((o.context, o.predict, o.device), (32, 7, 0));
        for args in [
            vec!["-c", "0"],
            vec!["-n", "-1"],
            vec!["--device", "metal"],
            vec!["--dtype", "bf16"],
            vec!["-m"],
            vec!["--bogus"],
        ] {
            assert!(parse(&args).is_err());
        }
        assert!(parse(&["--help"]).unwrap().is_none());
    }
}
