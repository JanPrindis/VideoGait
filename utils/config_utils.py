import os
import yaml
from skeletons import get_skeleton_by_name

def resolve_skeleton_from_config(app_config):
    """
    Determines the correct skeleton definition to use based on the application configuration.

    If the method is 'Heuristic', the skeleton is defined directly in the app config.
    If the method is 'NeuralNet', the skeleton is defined in the training config of the selected model.

    Args:
        app_config (dict): The loaded application configuration dictionary.

    Returns:
        SkeletonDefinition: The resolved skeleton definition object.
    """
    detector_cfg = app_config.get('event_detector', {})
    method = detector_cfg.get('method', 'Heuristic')
    skel_name = "halpe"  # Fallback default

    if method == 'Heuristic':
        # event_detector -> heuristic -> skeleton
        skel_name = detector_cfg.get('heuristic', {}).get('skeleton', 'halpe')

    elif method == 'NeuralNet':
        # event_detector -> neural_net -> experiment_path -> (load yaml) -> data -> skeleton
        exp_path = detector_cfg.get('neural_net', {}).get('experiment_path', '')
        nn_cfg_path = os.path.join(exp_path, 'config.yaml')

        if os.path.exists(nn_cfg_path):
            try:
                with open(nn_cfg_path, 'r') as f:
                    nn_config = yaml.safe_load(f)
                    skel_name = nn_config.get('data', {}).get('skeleton', 'halpe')
            except Exception as e:
                print(f"[ConfigUtils] Error loading NN config at {nn_cfg_path}: {e}")
        else:
            print(f"[ConfigUtils] Warning: NN config not found at {nn_cfg_path}, using default.")

    return get_skeleton_by_name(skel_name)
