import argparse
import torch
import sys

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.octmnist

import datasets.lungcolonv2
import datasets.chmnistv2
import datasets.covidv2
import datasets.btmriv2
import datasets.ctkidneyv2
import datasets.kvasirv2
import datasets.retinav2
import datasets.octmnistv2

import trainers.CoOp.coop_clip
import trainers.CoOp.coop_biomedclip
import trainers.CoOp.coop_pubmedclip
import trainers.CoOp.coop_pmcclip
import trainers.CoCoOp.cocoop_clip
import trainers.CoCoOp.cocoop_biomedclip
import trainers.CoCoOp.cocoop_pubmedclip
import trainers.CoCoOp.cocoop_pmcclip
import trainers.KgCoOp.kgcoop_clip
import trainers.KgCoOp.kgcoop_biomedclip
import trainers.KgCoOp.kgcoop_pubmedclip
import trainers.KgCoOp.kgcoop_pmcclip
import trainers.BiomedCoOp.biomedcoop_clip
import trainers.BiomedCoOp.biomedcoop_biomedclip
import trainers.BiomedCoOp.biomedcoop_pubmedclip
import trainers.BiomedCoOp.biomedcoop_pmcclip
import trainers.MeDKCoOp.medkcoop
import trainers.DCPL.dcpl
import trainers.DCPL.dcpl_biomedclip
import trainers.DCPL.dcpl_pmcclip
import trainers.DCPL.dcpl_pubmedclip
import trainers.GraphAdapter.baseclip_graph
import trainers.GraphAdapter.baseclip_graph_pmcclip
import trainers.MaPLe.maple


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.device_id:
        cfg.TRAINER.device_id = args.device_id

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.mode:
        cfg.MODEL.mode = args.mode



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # class-specific context
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50

    # Config for GRAPHDCPL
    cfg.TRAINER.GRAPHDCPL = CN()
    cfg.TRAINER.GRAPHDCPL.N_CTX = 2  # number of context vectors
    cfg.TRAINER.GRAPHDCPL.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.GRAPHDCPL.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.GRAPHDCPL.PREC_CLIP = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GRAPHDCPL.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.GRAPHDCPL.N_PROMPTS = 50
    cfg.TRAINER.GRAPHDCPL.hidden_dim = 512
    cfg.TRAINER.GRAPHDCPL.W = 1.0

    # Config for DCPL
    cfg.TRAINER.DCPL = CN()
    cfg.TRAINER.DCPL.N_CTX = 2  # number of context vectors
    cfg.TRAINER.DCPL.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.DCPL.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.DCPL.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)

    cfg.TRAINER.DPLQ = CN()
    cfg.TRAINER.DPLQ.N_CTX = 2  # number of context vectors
    cfg.TRAINER.DPLQ.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.DPLQ.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.DPLQ.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)

    # Config for GRAPHADAPTER
    cfg.TRAINER.GRAPHADAPTER = CN()
    cfg.TRAINER.GRAPHADAPTER.N_CTX = 2  # number of context vectors
    cfg.TRAINER.GRAPHADAPTER.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.GRAPHADAPTER.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GRAPHADAPTER.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.GRAPHADAPTER.N_PROMPTS = 50
    cfg.TRAINER.GRAPHADAPTER.hidden_dim = 1024
    cfg.TRAINER.GRAPHADAPTER.BETA = 0.3
    

    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.KGCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.W = 8

    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROGRAD.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROGRAD.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.PROGRAD.GM = False
    cfg.TRAINER.PROGRAD.NAME = ""
    cfg.TRAINER.PROGRAD.ALPHA = 0.
    cfg.TRAINER.PROGRAD.T = 1.
    cfg.TRAINER.PROGRAD.LAMBDA = 1.

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        if args.heatmap:
            trainer.test1()
        else:
            trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--device-id", type=int, help="gpu id"
    )
    parser.add_argument(
        "--mode", type=str, default="", help="train or test"
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--heatmap", type=str, default="", help="heatmap")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)