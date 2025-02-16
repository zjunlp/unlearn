import hydra
from src import it_unlearn


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    it_unlearn(cfg)

if __name__ == "__main__":
    main()
