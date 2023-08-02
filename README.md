# llama2 in Rust!

This is a Rust port of `https://github.com/karpathy/llama2.c`.

## Usage

```bash

# Clone the repo
git clone https://github.com/poudels14/llama2_rs

# Download the model
wget -P models https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# Run the inference

cargo run --release -- -t tokenizer.bin models/stories15M.bin 0.9

# For help, run:
cargo run --release -- --help

```

## License

MIT
