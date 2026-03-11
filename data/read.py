import h5py
import numpy as np

file_path = '/Users/kaelynwang/Desktop/kaelynwang/RLinf/data/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'

def print_training_data(name, obj):
    """
    Only prints datasets that are typically used for training:
    - actions
    - observations (images, states)
    - rewards
    """
    # Skip attributes entirely to avoid the messy XML output
    
    if isinstance(obj, h5py.Group):
        # We only care about the structure, so just pass
        pass
            
    elif isinstance(obj, h5py.Dataset):
        # Check if this dataset is relevant for training
        # Relevant paths usually end with: /actions, /rewards, /dones, or are inside /obs/
        
        is_relevant = False
        if name.endswith('/actions'):
            print(f"\n  [Actions] {name}")
            is_relevant = True
        elif '/obs/' in name:
            print(f"  [Observation] {name}")
            is_relevant = True
        elif name.endswith('/rewards'):
            print(f"  [Reward] {name}")
            is_relevant = True
            
        if is_relevant:
            print(f"    Shape: {obj.shape}")
            print(f"    Dtype: {obj.dtype}")
            
            # Print a clean data preview
            try:
                if len(obj.shape) >= 3:
                    # It's likely an image (T, H, W, C)
                    print(f"    Data: <Image Sequence> (min: {np.min(obj[0])}, max: {np.max(obj[0])})")
                elif len(obj.shape) == 2:
                    # Vector sequence (T, D)
                    print(f"    Data (first row): {obj[0]}")
                else:
                    # Scalar sequence (T,)
                    print(f"    Data (first 5): {obj[:]}")
            except Exception as e:
                print(f"    Could not print data preview: {e}")

try:
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("="*50)
        print("TRAINING DATA PREVIEW (Metadata hidden)")
        print("="*50)
        
        if 'data' in f:
            data_group = f['data']
            
            # Get sorted demo keys
            demo_keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else x)
            
            # Show only first 2 demos
            demos_to_show = demo_keys[:10]
            
            for demo_name in demos_to_show:
                print(f"\n>>> DEMO: {demo_name}")
                demo_group = data_group[demo_name]
                
                # Manually walk through specific interesting parts to keep output ordered and clean
                
                # 1. Actions (Output)
                if 'actions' in demo_group:
                    print_training_data(f"{demo_name}/actions", demo_group['actions'])
                
                # 2. Observations (Input)
                if 'obs' in demo_group:
                    print(f"\n  -- Observations (Inputs) --")
                    obs_group = demo_group['obs']
                    for obs_name, obs_ds in obs_group.items():
                        print_training_data(f"{demo_name}/obs/{obs_name}", obs_ds)
                
                # 3. Rewards
                if 'rewards' in demo_group:
                    print(f"\n  -- Labels/Feedback --")
                    print_training_data(f"{demo_name}/rewards", demo_group['rewards'])

        else:
            print("No 'data' group found.")

except Exception as e:
    print(f"Error reading file: {e}")
