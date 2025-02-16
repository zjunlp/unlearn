import hydra
from src import finetune


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    finetune(cfg)

if __name__ == "__main__":
    main()
