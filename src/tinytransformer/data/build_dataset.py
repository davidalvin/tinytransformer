import subprocess
import sys

def run(module_name):
    print(f"\n🚀 Running {module_name} ...\n")
    result = subprocess.run([sys.executable, "-m", module_name])
    if result.returncode != 0:
        raise RuntimeError(f"❌ Script failed: {module_name}")
    print(f"✅ Finished {module_name}")

def main():
    run("tinytransformer.data.prepare_dataset")     # Download and cache TinyStories
    run("tinytransformer.data.train_tokenizer")     # Train BPE tokenizer
    run("tinytransformer.data.save_tokens")         # Encode + save tokens

if __name__ == "__main__":
    main()
