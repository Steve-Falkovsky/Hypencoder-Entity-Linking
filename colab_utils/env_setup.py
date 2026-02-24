import os, subprocess, sys

TRANSFORMERS = "4.50.0"
SENTENCE_TRANSFORMERS = "2.2.2"

def install_env():
    print("This should take 20-30 seconds")
    print(f"Installing transformers=={TRANSFORMERS} and sentence_transformers=={SENTENCE_TRANSFORMERS}...")
    packages = [
        f"transformers=={TRANSFORMERS}",
        f"sentence_transformers=={SENTENCE_TRANSFORMERS}"
    ]
    
    # Run pip install
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + packages)
    print("Installed. Restarting session is required.")