from transformers import CLIPTokenizer
import argparse
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"
def load_tokenizer(args: argparse.Namespace):
    print("prepare tokenizer")
    original_path = V2_STABLE_DIFFUSION_PATH if args.v2 else TOKENIZER_PATH

    tokenizer: CLIPTokenizer = None
    if args.tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            print(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)  # same for v1 and v2

    if tokenizer is None:
        if args.v2:
            tokenizer = CLIPTokenizer.from_pretrained(original_path, subfolder="tokenizer")
        else:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        print(f"update token length: {args.max_token_length}")

    if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
        print(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer
