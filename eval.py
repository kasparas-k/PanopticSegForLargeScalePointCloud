from itertools import chain
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    
    test_root = (Path(cfg.data.dataroot) / cfg.data.dataset_name / 'raw/test').resolve()
    pcs = []
    for fmt in ['las', 'laz', 'ply']:
        pcs.extend(list(test_root.glob(f'**/*.{fmt}')))

    # test_root = (Path(cfg.data.dataroot) / cfg.data.dataset_name / 'raw').resolve()
    # pcs = []
    # for fmt in ['las', 'laz', 'ply']:
    #     pcs.extend(list(test_root.glob(f'**/*val.{fmt}')))
    cfg.data.fold = sorted(str(p) for p in pcs)[-1:]
    trainer = Trainer(cfg, True)
    trainer.eval(stage_name = "test")
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
