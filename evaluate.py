import os

# Force NLTK to use project-local data directory in both main process and
# multiprocessing spawn workers (prevents broken user-level punkt paths).
os.environ.setdefault("NLTK_DATA", os.path.join(os.path.dirname(__file__), "nltk_data"))

from lvlm.LLaVA import LLaVA
import torch
import argparse
import multiprocessing
from tqdm import tqdm
import json 
import random
import os
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from PIL import Image
from util.chair import CHAIR
from modelscope import snapshot_download

LVLM_MAP = {
    'llava-1.5-13b-hf': LLaVA,
    'llava-1.5-7b-hf': LLaVA,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvlm', type=str, default='llava-1.5-7b-hf')
    parser.add_argument('--dataset', type=str, default='MSCOCO')
    parser.add_argument('--inference_temp', type=float, default=0.1)
    parser.add_argument('--sampling_temp', type=float, default=1.0)
    parser.add_argument('--sampling_time', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--generate', type=bool, default=True)
    parser.add_argument('--num_data', type=int, default=500)
    parser.add_argument('--image_layer', type=int, default=32)
    parser.add_argument('--text_layer', type=int, default=31)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--w', type=float, default=0.6)
    parser.add_argument('--num_gpus', type=int, default=1, help="使用几张卡并行；1 则单卡顺序跑")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def obtain_lvlm(args):
    lvlm_class = LVLM_MAP.get(args.lvlm)
    if not lvlm_class:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")

    return lvlm_class(args.lvlm)

def extract_tensors(data_dict):
    """Extracts tensors from a dictionary and converts them to a NumPy array."""
    tensor_list = []
    for obj, tensor_list_per_obj in data_dict.items():
        tensor_list.extend([t.cpu().numpy() for t in tensor_list_per_obj])  # Convert tensors to NumPy
    return np.array(tensor_list)  # Shape: (num_samples, 33)

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_weighted_scores(global_dict, topk_dict, args):
    """
    把每个 token 对应的 (33x33) tensor 压缩成一个标量分数：
    score = w*global[text_layer, image_layer] + (1-w)*topk[text_layer, image_layer]
    这样多进程只回传标量，避免 Queue 传大数组导致汇总卡住。
    """
    scores = []
    for k, g_list in global_dict.items():
        t_list = topk_dict.get(k, [])
        n = min(len(g_list), len(t_list))
        for i in range(n):
            g = g_list[i]
            t = t_list[i]
            s = args.w * float(g[args.text_layer, args.image_layer].item()) + (1 - args.w) * float(
                t[args.text_layer, args.image_layer].item()
            )
            scores.append(s)
    return scores


def _run_one_gpu(gpu_id, entries, args_dict, question, mscoco_val_dir, q):
    # 每个进程固定到指定 GPU，并各自只加载一次模型
    try:
        torch.cuda.set_device(gpu_id)
        args = argparse.Namespace(**args_dict)

        lvlm = LLaVA(args.lvlm, model_dir=args_dict["model_dir"], device=gpu_id)

        true_scores_all = []
        false_scores_all = []
        for entry in tqdm(entries, desc=f"GPU {gpu_id}", position=gpu_id):
            image_path = os.path.join(mscoco_val_dir, entry["image"])
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path).convert("RGB")
            result = lvlm.generate(image, question, entry["image_id"], args)

            true_scores_all.extend(
                _extract_weighted_scores(result["global_cos_matrix_true"], result["top_k_cos_matrix_true"], args)
            )
            false_scores_all.extend(
                _extract_weighted_scores(result["global_cos_matrix_false"], result["top_k_cos_matrix_false"], args)
            )

        q.put((gpu_id, np.array(true_scores_all, dtype=np.float32), np.array(false_scores_all, dtype=np.float32), None))
    except Exception as e:
        # 把错误回传给主进程，避免主进程一直卡在 q.get()
        q.put((gpu_id, None, None, repr(e)))

def main():
    fix_seed(0)
    args = parse_args()
    if args.dataset == "MSCOCO":
        MSCOCO_VAL_DIR = "/home/apulis-dev/userdata/val2014"
        COCO_ANNOTATION_PATH = "/home/apulis-dev/userdata/annotations/captions_val2014.json"
          
    QUESTION = "Describe the given image in detail."
    
    # # Load COCO annotations
    # with open(COCO_ANNOTATION_PATH, "r") as f:
    #     coco_data = [json.loads(line) for line in f]

    # Load COCO annotations (适配标准 COCO 格式)
    with open(COCO_ANNOTATION_PATH, "r") as f:
        content = json.load(f)  # 加载整个 JSON
        
        coco_data = []
        # 标准 COCO 格式中，图片信息存储在 'images' 键下
        if 'images' in content:
            for img in content['images']:
                coco_data.append({
                    "image": img['file_name'],  # 把 file_name 映射为代码需要的 image
                    "image_id": img['id']
                })
        else:
            # 防御性代码：万一你是另一种格式（列表格式）
            coco_data = content
    
    coco_gt = random.sample(coco_data, args.num_data)
    
    if args.generate == True:
        requested_gpus = max(1, args.num_gpus)
        available_gpus = torch.cuda.device_count()
        num_gpus = min(requested_gpus, available_gpus, len(coco_gt))

        # 统一下载/定位模型目录，只执行一次
        model_dir = snapshot_download(
            'llava-hf/llava-1.5-7b-hf',
            cache_dir='/home/apulis-dev/models/llava',
        )
        args_dict = vars(args).copy()
        args_dict["model_dir"] = model_dir

        # 切分数据到每张卡
        if num_gpus <= 1:
            torch.cuda.set_device(0)
            lvlm = LLaVA(args.lvlm, model_dir=model_dir, device=0)
            true_scores_all = []
            false_scores_all = []
            for entry in tqdm(coco_gt, desc="Processing Images"):
                image_filename = entry["image"]
                image_path = os.path.join(MSCOCO_VAL_DIR, image_filename)

                if not os.path.exists(image_path):
                    continue

                image = Image.open(image_path).convert("RGB")
                result = lvlm.generate(image, QUESTION, entry["image_id"], args)

                true_scores_all.extend(
                    _extract_weighted_scores(result["global_cos_matrix_true"], result["top_k_cos_matrix_true"], args)
                )
                false_scores_all.extend(
                    _extract_weighted_scores(result["global_cos_matrix_false"], result["top_k_cos_matrix_false"], args)
                )

            true_scores = np.array(true_scores_all, dtype=np.float32)
            false_scores = np.array(false_scores_all, dtype=np.float32)
        else:
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()
            chunks = [[] for _ in range(num_gpus)]
            for i, entry in enumerate(coco_gt):
                chunks[i % num_gpus].append(entry)

            processes = []
            for gpu_id in range(num_gpus):
                if not chunks[gpu_id]:
                    continue
                p = ctx.Process(
                    target=_run_one_gpu,
                    args=(gpu_id, chunks[gpu_id], args_dict, QUESTION, MSCOCO_VAL_DIR, q),
                )
                p.start()
                processes.append(p)

            per_gpu = {}
            for _ in range(len(processes)):
                gpu_id, t_scores, f_scores, err = q.get()
                if err is not None:
                    raise RuntimeError(f"Worker GPU {gpu_id} failed: {err}")
                per_gpu[gpu_id] = (t_scores, f_scores)

            for p in processes:
                p.join()

            true_scores = np.concatenate([per_gpu[i][0] for i in sorted(per_gpu.keys()) if per_gpu[i][0].size > 0])
            false_scores = np.concatenate([per_gpu[i][1] for i in sorted(per_gpu.keys()) if per_gpu[i][1].size > 0])

                
        def compute_layerwise_metrics(true_scores, false_scores):
            N = true_scores.shape[0]
            M = false_scores.shape[0]
            y_true = np.concatenate([np.ones(N), np.zeros(M)])
            y_scores = np.concatenate([true_scores, false_scores])
            auroc = roc_auc_score(y_true, y_scores)
            aupr = average_precision_score(y_true, y_scores)
            return {"auroc": auroc, "aupr": aupr}
                
        metrics = compute_layerwise_metrics(true_scores, false_scores)

        # 评估结果：终端打印
        print("\n========== Evaluation Results ==========")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPR:  {metrics['aupr']:.4f}")
        print("========================================\n")

        # 可选：保存到 JSON（便于记录/复现）
        results_path = "evaluation_results.json"

        # 构造当前实验记录（包含时间和数据规模信息）
        current_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "num_data": args.num_data,
            "metrics": metrics,
        }

        # 如果文件已存在，则读出并在末尾追加；否则创建新列表
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

            # 兼容老格式：如果之前是单个 dict，就转成列表
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []

            existing.append(current_record)
        else:
            existing = [current_record]

        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        print(f"Results appended to {results_path}")

if __name__ == "__main__":
    main()