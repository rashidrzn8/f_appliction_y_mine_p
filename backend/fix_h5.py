import os
import h5py
import json

def clean_h5(filepath):
    print(f"Cleaning {filepath} ...")
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                config_json = f.attrs['model_config']
                if isinstance(config_json, bytes):
                    config_str = config_json.decode('utf-8')
                else:
                    config_str = config_json
                
                config = json.loads(config_str)
                
                # Traverse and remove 'quantization_config'
                def remove_qconfig(obj):
                    if isinstance(obj, dict):
                        if 'quantization_config' in obj:
                            del obj['quantization_config']
                        for k, v in obj.items():
                            remove_qconfig(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_qconfig(item)
                            
                remove_qconfig(config)
                
                new_config_str = json.dumps(config)
                if isinstance(config_json, bytes):
                    f.attrs['model_config'] = new_config_str.encode('utf-8')
                else:
                    f.attrs['model_config'] = new_config_str
                print("Successfully cleaned model_config in", filepath)
            else:
                print("No model_config found in", filepath)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == '__main__':
    for file in os.listdir('.'):
        if file.endswith('.h5'):
            clean_h5(file)
