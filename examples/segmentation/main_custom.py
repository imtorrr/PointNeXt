import sys
import os

sys.path.append(os.path.abspath("."))


import argparse
import yaml
import os
import logging
import numpy as np
import csv
import wandb
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import (
    set_random_seed,
    save_checkpoint,
    load_checkpoint,
    resume_checkpoint,
    setup_logger_dist,
    cal_model_parm_nums,
    Wandb,
    generate_exp_directory,
    resume_exp_directory,
    EasyConfig,
    dist_utils,
    find_free_port,
    load_checkpoint_inv,
)
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import (
    build_dataloader_from_cfg,
    get_features_by_keys,
    get_class_weights,
)
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import (
    load_label_kitti,
    load_pc_kitti,
    remap_lut_read,
    remap_lut_write,
    get_semantickitti_file_list,
)
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f"{item:.2f}" for item in ious]
    header = (
        ["method", "Area", "OA", "mACC", "mIoU"]
        + cfg.classes
        + ["best_epoch", "log_path", "wandb link"]
    )
    data = (
        [cfg.cfg_basename, str(area), f"{oa:.2f}", f"{macc:.2f}", f"{miou:.2f}"]
        + ious_table
        + [
            str(best_epoch),
            cfg.run_dir,
            wandb.run.get_url() if cfg.wandb.use_wandb else "-",
        ]
    )
    with open(cfg.csv_path, "a", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()
        
def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get("in_channels", None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info("Number of params: %.4f M" % (model_size / 1e6))
    
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info("Using Synchronized BatchNorm ...")
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank
        )
        logging.info("Using Distributed Data parallel ...")
        
    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)
    
    # build dataset
    val_loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.batch_size),
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="val",
        distributed=cfg.distributed,
    )
    
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = (
        val_loader.dataset.num_classes
        if hasattr(val_loader.dataset, "num_classes")
        else None
    )
    
    if num_classes is not None:
        assert cfg.num_classes == num_classes
        
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = (
        val_loader.dataset.classes
        if hasattr(val_loader.dataset, "classes")
        else np.arange(num_classes)
    )
    cfg.cmap = (
        np.array(val_loader.dataset.cmap)
        if hasattr(val_loader.dataset, "cmap")
        else None
    )
    
    validate_fn = validate
    
    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, "module") else model
    if cfg.pretrained_path is not None:
        if cfg.mode == "resume":
            resume_checkpoint(
                cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path
            )
        else:
            if cfg.mode == "val":
                best_epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path
                )
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(
                    model, val_loader, cfg, num_votes=1, epoch=epoch
                )
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f"Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, "
                        f"\niou per cls is: {val_ious}"
                    )
                return val_miou
            elif cfg.mode == "test":
                raise ValueError("Test mode is not supported")

            elif "encoder" in cfg.mode:
                if "inv" in cfg.mode:
                    logging.info(f"Finetuning from {cfg.pretrained_path}")
                    load_checkpoint_inv(model.encoder, cfg.pretrained_path)
                else:
                    logging.info(f"Finetuning from {cfg.pretrained_path}")
                    load_checkpoint(
                        model_module.encoder,
                        cfg.pretrained_path,
                        cfg.get("pretrained_module", None),
                    )

            else:
                logging.info(f"Finetuning from {cfg.pretrained_path}")
                load_checkpoint(
                    model, cfg.pretrained_path, cfg.get("pretrained_module", None)
                )
    else:
        logging.info("Training from scratch")
        
    if "freeze_blocks" in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False
            
    train_loader = build_dataloader_from_cfg(
        cfg.batch_size,
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="train",
        distributed=cfg.distributed,
    )
    
    
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    cfg.criterion_args.weight = None
    if cfg.get("cls_weighed_loss", False):
        if hasattr(train_loader.dataset, "num_per_class"):
            cfg.criterion_args.weight = get_class_weights(
                train_loader.dataset.num_per_class, normalize=True
            )
        else:
            logging.info("`num_per_class` attribute is not founded in dataset")
            
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    
    # ===> start training
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    val_miou, val_macc, val_oa, val_ious, val_accs = 0.0, 0.0, 0.0, [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = (
        0.0,
        0.0,
        0.0,
        [],
        0,
    )
    total_iter = 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(
            train_loader.dataset, "epoch"
        ):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_miou, train_macc, train_oa, _, _, total_iter = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                epoch,
                total_iter,
                cfg,
            )
        )

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(
                model, val_loader, cfg, epoch=epoch, total_iter=total_iter
            )
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f"Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}"
                        f"\nmious: {val_ious}"
                    )

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch} LR {lr:.6f} "
            f"train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}"
        )
        if writer is not None:
            writer.add_scalar("best_val", best_val, epoch)
            writer.add_scalar("val_miou", val_miou, epoch)
            writer.add_scalar("macc_when_best", macc_when_best, epoch)
            writer.add_scalar("oa_when_best", oa_when_best, epoch)
            writer.add_scalar("val_macc", val_macc, epoch)
            writer.add_scalar("val_oa", val_oa, epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_miou", train_miou, epoch)
            writer.add_scalar("train_macc", train_macc, epoch)
            writer.add_scalar("lr", lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(
                cfg,
                model,
                epoch,
                optimizer,
                scheduler,
                additioanl_dict={"best_val": best_val},
                is_best=is_best,
            )
            is_best = False
    
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f"Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, "
            f"\niou per cls is: {ious_when_best}"
        )
    
    if cfg.world_size < 2:  # do not support multi gpu testing
        # test
        load_checkpoint(
            model,
            pretrained_path=os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_ckpt_best.pth"),
        )
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + ".csv")
        if "sphere" in cfg.dataset.common.NAME.lower():
            # TODO:
            test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(
                model, val_loader, cfg, epoch=epoch
            )
        else:
            data_list = generate_data_list(cfg)
            test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(
                model, data_list, cfg
            )
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f"Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, "
                f"\niou per cls is: {test_ious}"
            )
        if writer is not None:
            writer.add_scalar("test_miou", test_miou, epoch)
            writer.add_scalar("test_macc", test_macc, epoch)
            writer.add_scalar("test_oa", test_oa, epoch)
        write_to_csv(
            test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True
        )
        logging.info(f"save results in {cfg.csv_path}")
        if cfg.use_voting:
            load_checkpoint(
                model,
                pretrained_path=os.path.join(
                    cfg.ckpt_dir, f"{cfg.run_name}_ckpt_best.pth"
                ),
            )
            set_random_seed(cfg.seed)
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(
                model,
                val_loader,
                cfg,
                num_votes=20,
                data_transform=data_transform,
                epoch=epoch,
            )
            if writer is not None:
                writer.add_scalar("val_miou20", val_miou, cfg.epochs + 50)

            ious_table = [f"{item:.2f}" for item in val_ious]
            data = (
                [
                    cfg.cfg_basename,
                    "True",
                    f"{val_oa:.2f}",
                    f"{val_macc:.2f}",
                    f"{val_miou:.2f}",
                ]
                + ious_table
                + [str(best_epoch), cfg.run_dir]
            )
            with open(cfg.csv_path, "w", encoding="UT8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
    else:
        logging.warning(
            "Testing using multiple GPUs is not allowed for now. Running testing after this training is required."
        )
    if writer is not None:
        writer.close()
    # dist.destroy_process_group() # comment this line due to https://github.com/guochengqian/PointNeXt/issues/95
    wandb.finish(exit_code=True)
    
def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg
):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        import pdb;pdb.set_trace()
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data["y"].squeeze(-1)
        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        data["x"] = get_features_by_keys(data, cfg.feature_keys)
        data["epoch"] = epoch
        total_iter += 1
        data["iter"] = total_iter
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(data)
            loss = (
                criterion(logits, target)
                if "mask" not in cfg.criterion_args.NAME.lower()
                else criterion(logits, target, data["mask"])
            )

        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get("grad_norm_clip") is not None and cfg.grad_norm_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2
                )
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)
            # mem = torch.cuda.max_memory_allocated() / 1024. / 1024.
            # print(f"Memory after backward is {mem}")

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(
                f"Train Epoch [{epoch}/{cfg.epochs}] "
                f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}"
            )
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs, total_iter


@torch.no_grad()
def validate(
    model, val_loader, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1
):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc="Val")
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data["y"].squeeze(-1)
        data["x"] = get_features_by_keys(data, cfg.feature_keys)
        data["epoch"] = epoch
        data["iter"] = total_iter
        logits = model(data)
        if "mask" not in cfg.criterion_args.NAME or cfg.get("use_maks", False):
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data["mask"].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        coord = data['pos'].cpu().numpy()[0]
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        label = target[0].cpu().numpy()
        if cfg.ignore_index is not None:
            if (label == cfg.ignore_index).sum() > 0:
                pred[label == cfg.ignore_index] = cfg.num_classes
                label[label == cfg.ignore_index] = cfg.num_classes
        vis_multi_points([coord, coord], labels=[label, pred])
        """
        # tp, union, count = cm.tp, cm.union, cm.count
        # if cfg.distributed:
        #     dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        # miou, macc, oa, ious, accs = get_mious(tp, union, count)
        # with np.printoptions(precision=2, suppress=True):
        #     logging.info(f'{idx}-th cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
        #                 f'\niou per cls is: {ious}')

    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scene segmentation training/testing")
    parser.add_argument("--cfg", type=str, required=True, help="config file")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="set to True to profile speed",
    )
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split(".")[-2].split("/")[
        -2
    ]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split(".")[-2].split("/")[
        -1
    ]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f"ngpus{cfg.world_size}",
    ]
    opt_list = []  # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if (
            "rank" not in opt
            and "dir" not in opt
            and "root" not in opt
            and "pretrain" not in opt
            and "path" not in opt
            and "wandb" not in opt
            and "/" not in opt
        ):
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = "-".join(opt_list)

    cfg.is_training = cfg.mode not in ["test", "testing", "val", "eval", "evaluation"]
    if cfg.mode in ["resume", "val", "test"]:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(
            cfg, tags, additional_id=os.environ.get("MASTER_PORT", None)
        )
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, indent=2)
        os.system("cp %s %s" % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print("using mp spawn for distributed training")
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)