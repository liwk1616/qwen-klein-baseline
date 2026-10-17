"""Score every planned story target; blocked or missing outputs count as failures."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
from archive_io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classifiers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-label", type=str)
    args = parser.parse_args()
    os.environ.update(TORCH_COMPILE_DISABLE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from torchvision.models import inception_v3
    from torchvision import transforms
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    outputs = [json.loads(line) for path in sorted(args.generation.glob('rank_*/generated.jsonl')) for line in path.read_text().splitlines() if line.strip()]
    outputs.sort(key=lambda row: row["index"])
    if len(outputs) != len(rows) or [o["example_id"] for o in outputs] != [row["example_id"] for row in rows] or [o["clip_id"] for o in outputs] != [row["clip_id"] for row in rows]:
        raise ValueError("Generation records do not cover the complete planned evaluation")
    with (args.data / "labels.pkl").open("rb") as stream:
        characters = pickle.load(stream)
    with (args.data / "labels_bg.pkl").open("rb") as stream:
        backgrounds = pickle.load(stream)
    labels = {"char": torch.tensor([characters[row["clip_id"]] for row in rows], dtype=torch.int64), "bg": torch.tensor([backgrounds["label"][row["clip_id"]] for row in rows], dtype=torch.int64)}
    networks = {}
    for mode, classes in (("char", 7), ("bg", 323)):
        network = inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False, num_classes=classes)
        network.load_state_dict(load_file(args.classifiers / f"classifier_{mode}.safetensors"), strict=True)
        networks[mode] = network.eval().requires_grad_(False).to("cuda:0")
    preprocess = transforms.Compose([transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), transforms.CenterCrop(299), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    available = [index for index, row in enumerate(outputs) if row["status"] == "complete"]
    logits = {mode: [] for mode in networks}
    with torch.inference_mode():
        for start in range(0, len(available), 16):
            images = []
            for index in available[start:start + 16]:
                with Image.open(outputs[index]["image"]) as image:
                    if image.size != (512, 512) or image.format != "BMP":
                        raise ValueError("Unexpected generated image")
                    images.append(preprocess(image.convert("RGB")))
            inputs = torch.stack(images).to("cuda:0")
            for mode, network in networks.items():
                logits[mode].append(network(inputs).float().cpu())
    predicted = {mode: torch.zeros_like(label) for mode, label in labels.items()}
    tensors = {"successful_global_indices": torch.tensor(available, dtype=torch.int64)}
    for mode in networks:
        values = torch.cat(logits[mode]) if available else torch.empty((0, labels[mode].shape[1]), dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Non-finite classifier logits")
        tensors[mode + "_logits_successful_only"] = values
        if available:
            predicted[mode][available] = torch.round(torch.sigmoid(values)).to(torch.int64)
    save_file(tensors, args.output / "classifier_logits.safetensors")
    successful = torch.zeros(len(rows), dtype=torch.bool)
    successful[available] = True
    scores = {}
    exacts = {}
    for mode, value in predicted.items():
        target = labels[mode]
        exact = (value == target).all(dim=1) & successful
        exacts[mode] = exact
        true_positive = int(((value == 1) & (target == 1)).sum())
        denominator = int(value.sum()) + int(target.sum())
        scores[mode] = {"accuracy_percent": 100 * int(exact.sum()) / len(rows), "exact_matches": int(exact.sum()), "planned_examples": len(rows), "generated_examples": len(available), "accuracy_on_generated_only_percent": 100 * int(exact.sum()) / len(available) if available else None, "micro_f1_percent": 200 * true_positive / denominator if denominator else 0.0}
    with (args.output / "predictions.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            record = {"index": index, "example_id": row["example_id"], "clip_id": row["clip_id"], "task_id": row["task_id"], "generation_status": outputs[index]["status"], "image": outputs[index].get("image")}
            for mode in networks:
                record[mode + "_target_indices"] = labels[mode][index].nonzero().flatten().tolist()
                record[mode + "_predicted_indices"] = predicted[mode][index].nonzero().flatten().tolist() if successful[index] else None
                record[mode + "_exact_match"] = bool(exacts[mode][index])
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"status": "complete", "task": "source-constrained story continuation pilot", "scores": scores, "planned_targets": len(rows), "generated_targets": len(available), "blocked_or_missing_targets": len(rows) - len(available), "source_status_counts": {status: sum(row["status"] == status for row in outputs) for status in sorted({row["status"] for row in outputs})}, "failure_handling": "No-output targets count as incorrect for both accuracy metrics, including empty target label sets", "formal_contract_benchmark": False, "manifest": str(args.manifest.resolve()), "generation": str(args.generation.resolve())}
    story_groups = [[row for row in outputs if row["task_id"] == task] for task in dict.fromkeys(row["task_id"] for row in outputs)]
    summary.update(planned_stories=len(story_groups), fully_generated_stories=sum(all(row["status"] == "complete" for row in group) for group in story_groups), stories_with_any_generated_target=sum(any(row["status"] == "complete" for row in group) for group in story_groups))
    source_gate_modes = sorted({row.get("source_gate", "block") for row in outputs})
    summary.update(source_gate_modes=source_gate_modes, source_gate_bypassed_targets=sum(row.get("source_gate_bypassed", False) for row in outputs), generation_success_is_not_identity_acceptance=True)
    if "report-only" in source_gate_modes:
        summary["task"] = "story continuation diagnostic with source admission reported but not enforced"
    if args.task_label:
        summary['task'] = args.task_label
    atomic_json(args.output / "scores.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
