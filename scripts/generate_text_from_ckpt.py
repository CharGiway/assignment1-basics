import argparse
import json
from pathlib import Path
import torch
from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.optim.adamw import AdamW
from cs336_basics.serialization import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--vocab_path", type=str, required=True)
    ap.add_argument("--merges_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Once upon a time, there was a curious child named Mia who")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    # model arch (must match training)
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=1344)
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    args = ap.parse_args()

    device = torch.device(args.device)

    tok = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    eot_id = tok.encode("<|endoftext|>")[0]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    model.to(device)
    optim = AdamW(model.parameters(), lr=1e-4)
    _ = load_checkpoint(args.checkpoint_path, model, optim)
    model.to(device)
    model.eval()

    prompt_ids = torch.tensor(tok.encode(args.prompt), dtype=torch.long)
    out_ids = decode(
        model,
        prompt_ids,
        eot_id=eot_id,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        device=device,
        return_full_sequence=True,
    )
    out_text = tok.decode(list(map(int, out_ids.tolist())))

    print(json.dumps({
        "prompt": args.prompt,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "checkpoint_path": args.checkpoint_path,
        "out_text": out_text,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
