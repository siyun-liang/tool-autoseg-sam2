import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder,VanillaVAE


def resolve_feature_files(dataset_path, language_name, levels):
    files = []
    for lv in levels:
        lv_dir = os.path.join(dataset_path, lv, language_name)
        if not os.path.isdir(lv_dir):
            continue
        files.extend(sorted([os.path.join(lv_dir, x) for x in os.listdir(lv_dir) if x.endswith("_f.npy")]))
    return files


def output_root_for_input(input_f_path, dataset_path, language_name, out_dir_name):
    # aggregate layout: <dataset>/<level>/<language_name>/<frame>_f.npy
    abs_dataset = os.path.abspath(dataset_path)
    abs_input = os.path.abspath(input_f_path)
    rel = os.path.relpath(abs_input, abs_dataset)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        level = parts[0]
        return os.path.join(abs_dataset, level, out_dir_name)
    return os.path.join(abs_dataset, out_dir_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--hidden_dims',
                    type=int,
                    default=3
                    )
    parser.add_argument('--feature_dims',
                    type=int,
                    default=512
                    )
    # parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument('--language_name', type = str, default = None)
    parser.add_argument('--output_name',type=str,default='clip_features')
    parser.add_argument('--levels', type=str, default='default,small,middle,large')
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='_s.npy',
        help="Output filename suffix for encoded features, e.g. _s.npy or _f.npy",
    )
    args = parser.parse_args()
    
    model_name = args.model_name
    encoder_hidden_dims = args.encoder_dims
    # encoder_hidden_dims[-1] = args.output_dim
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_root = os.path.abspath(os.path.join(dataset_path, "..", "ckpt"))
    ckpt_path = os.path.join(ckpt_root, model_name, "best_ckpt.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    language_name = args.language_name or "language_features"
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    feature_files = resolve_feature_files(
        dataset_path=dataset_path,
        language_name=language_name,
        levels=levels,
    )
    if len(feature_files) == 0:
        raise RuntimeError(
            f"No *_f.npy files found for export under levels={levels} and language_name={language_name}."
        )

    out_dir_name = f"{args.output_name}_dim{encoder_hidden_dims[-1]}"


    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_names=feature_files)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )

    if os.getenv("use_vae",'f') == 't':
        model = VanillaVAE(encoder_hidden_dims, decoder_hidden_dims, latent_dim=9).to("cuda:0")
    else:
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, feature_dim=args.feature_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        data = data.to(torch.float32)
        
        with torch.no_grad():
            if os.getenv("use_vae",'f') == 't':
                mu, log_var  = model.encode(data)
                outputs = model.reparameterize(mu, log_var).to('cpu').numpy()  

            else:
                outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    start = 0

    for info in train_dataset.sample_info:
        in_path = info["path"]
        rows = int(info["rows"])
        src_name = os.path.basename(in_path)
        if src_name.endswith("_f.npy"):
            dst_name = src_name.replace("_f.npy", args.output_suffix)
        else:
            dst_name = os.path.splitext(src_name)[0] + args.output_suffix
        dst_root = output_root_for_input(
            input_f_path=in_path,
            dataset_path=dataset_path,
            language_name=language_name,
            out_dir_name=out_dir_name,
        )
        os.makedirs(dst_root, exist_ok=True)
        out_path = os.path.join(dst_root, dst_name)
        np.save(out_path, features[start:start + rows])
        start += rows
