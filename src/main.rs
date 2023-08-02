mod llama;
mod math;
mod reader;
mod transformer;
mod vocab;

use anyhow::Result;
use clap::Parser;
use llama::Config;
use llama::RunOptions;
use rayon::ThreadPoolBuilder;
use reader::FloatReader;
use std::fs::File;
use std::io::BufReader;
use vocab::Vocab;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    /// Path to the tokenizer model
    tokenizer: String,

    /// Path to the model weights
    model: String,

    /// Temperature
    #[arg(default_value_t = 0.9)]
    temperature: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file = File::open(args.model)?;
    let mut reader = BufReader::new(file);

    let mut r = FloatReader::new(&mut reader);
    let config: Config = llama::read_config(&mut r)?;
    let weights = llama::init_checkpoint_weights(&mut r, &config)?;
    let vocab = Vocab::from_file(config.vocab_size, &args.tokenizer);
    let mut state = llama::init_run_state(&config);

    let pool = ThreadPoolBuilder::new()
        // 2 threads seems to perform best
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        llama::run(
            &config,
            &mut state,
            &weights,
            &vocab,
            RunOptions {
                temperature: args.temperature,
            },
        );
    });

    Ok(())
}
