import os
import sys
import subprocess
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import dotenv
from utils.utils import get_logger, ensure_dir

# Load environment variables if exists
dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, "pc_environment.env")
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path, override=True)

def validate_config(cfg: DictConfig, mode: str) -> bool:
    """
    Validate configuration parameters for the specified mode
    
    Args:
        cfg: Configuration object
        mode: Operation mode (train, test, demo)
        
    Returns:
        Whether the configuration is valid
    """
    log = get_logger(__name__)
    
    # Check basic configuration
    if not hasattr(cfg, 'mode') or not hasattr(cfg.mode, mode):
        log.error(f"Configuration missing: cfg.mode.{mode}")
        return False
        
    # Check dataset configuration
    required_dataset_fields = ['data_root']
    if not hasattr(cfg, 'dataset'):
        log.error("Configuration missing: cfg.dataset")
        return False
    
    for field in required_dataset_fields:
        if not hasattr(cfg.dataset, field):
            log.error(f"Configuration missing: cfg.dataset.{field}")
            return False
    
    # Mode-specific validation
    if mode == 'train':
        if not hasattr(cfg.dataset, 'train_mat'):
            log.error("Configuration missing: cfg.dataset.train_mat")
            return False
    
    elif mode == 'test':
        if not hasattr(cfg, 'roi_detection'):
            log.error("Configuration missing: cfg.roi_detection")
            return False
        if not hasattr(cfg, 'classification'):
            log.error("Configuration missing: cfg.classification")
            return False
        if not hasattr(cfg.dataset, 'test_mat'):
            log.error("Configuration missing: cfg.dataset.test_mat")
            return False
    
    elif mode == 'demo':
        if not hasattr(cfg, 'roi_detection'):
            log.error("Configuration missing: cfg.roi_detection")
            return False
        if not hasattr(cfg, 'classification'):
            log.error("Configuration missing: cfg.classification")
            return False
    
    log.info("Configuration validation passed")
    return True

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Paris Metro Line Recognition system
    
    Args:
        cfg: Hydra configuration object
    """
    log = get_logger(__name__)
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Handle test mode (unit tests)
    if cfg.mode.get("test_mode", False):
        log.info("Test mode is enabled")
        test_dir = cfg.mode.get("test_dir", "tests")
        test_file = cfg.mode.get("test_file", "")

        if not os.path.exists(test_dir):
            log.error(f"Test directory does not exist: {test_dir}")
            return

        if test_file:
            test_path = os.path.join(test_dir, test_file)
            if not os.path.exists(test_path):
                log.error(f"Test file does not exist: {test_path}")
                return
            log.info(f"Executing specified test file: {test_path}")
            cmd = f"python -m unittest {test_path}"
        else:
            log.info(f"No test file specified, executing all tests in directory {test_dir}")
            cmd = f"python -m unittest discover {test_dir}"

        try:
            ret = subprocess.call(cmd, shell=True)
            log.info(f"Test completed, exit code {ret}")
        except Exception as e:
            log.error(f"Error executing tests: {e}")
        return

    # Get operation mode
    mode_name = cfg.mode.get("name", "")
    dispatch_dict = {
        "train": "src.pipeline.train_pipeline:main",
        "test": "src.pipeline.test_pipeline:main",
        "demo": "src.pipeline.demo_pipeline:main"
    }

    if not mode_name:
        log.error("Please specify mode.name in the configuration (train, test, or demo)")
        sys.exit(1)
        
    if mode_name not in dispatch_dict:
        log.error(f"Invalid mode: {mode_name}. Supported modes: {list(dispatch_dict.keys())}")
        sys.exit(1)
        
    # Validate configuration
    #if not validate_config(cfg, mode_name):
    #    log.error("Configuration validation failed, please check the configuration file")
    #    sys.exit(1)

    # Ensure output directory exists
    output_dir = cfg.get("output_dir", "results") 
    ensure_dir(output_dir)
    log.info(f"Output directory: {output_dir}")

    # Dynamic module import and execution
    import importlib
    path_func_str = dispatch_dict[mode_name] 
    module_path, func_name = path_func_str.split(":")

    try:
        log.info(f"Loading module: {module_path}")
        mod = importlib.import_module(module_path)
    except ImportError as e:
        log.error(f"Failed to import module {module_path}: {e}")
        sys.exit(1)
        
    try:
        main_func = getattr(mod, func_name)
        if not callable(main_func):
            log.error(f"{module_path}.{func_name} is not callable")
            sys.exit(1)
            
        log.info(f"Executing {path_func_str} in {mode_name} mode")
        main_func(cfg)
    except AttributeError as e:
        log.error(f"Function {func_name} not found in module {module_path}: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error executing {path_func_str}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 