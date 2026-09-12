"""Render fixed held-out examples, without cherry-picking by model error."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from pilot import dataset, make_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    batch, _ = dataset(24000, 64)
    examples = [0, 7, 31]
    columns = [(ep, h) for ep in examples for h in (1, 3, 5)]
    size, label = 96, 190
    canvas = Image.new("RGB", (label + len(columns)*size, 80+4*(size+28)), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Synthetic pilot: held-out future frames | width 16, seed 0, 1000 steps", fill="black")
    for col, (ep, horizon) in enumerate(columns):
        draw.text((label+col*size, 50), f"ep{ep} h{horizon}", fill="black")
    rows = [("Actual future", None), ("Compact LeWM", "compact_lewm"),
            ("Predictive, no SIGReg", "predictive"), ("Predictive + SIGReg", "predictive_sigreg")]
    for row, (title, variant) in enumerate(rows):
        y = 80 + row * (size + 28)
        draw.text((10, y+40), title, fill="black")
        predictions = {}
        if variant:
            m = make_model(16).eval()
            m.load_state_dict(torch.load(args.directory / f"{variant}-d16-s0.pt", weights_only=True, map_location="cpu"))
            out = m.encode(dict(batch))
            ctx = out["emb"][:, :3]
            for k in range(1, 6):
                pred = m.predict(ctx[:, -3:], out["act_emb"][:, k-1:k+2])[:, -1:]
                ctx = torch.cat((ctx, pred), 1)
                if k in (1, 3, 5):
                    predictions[k] = m.decoder(pred[:, 0])
        for col, (ep, k) in enumerate(columns):
            pixels = predictions[k][ep] if variant else batch["pixels"][ep, 2+k]
            rgb = (pixels.clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy()
            canvas.paste(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST), (label+col*size, y))
    canvas.save(args.directory / "predictions.png")


if __name__ == "__main__":
    main()
