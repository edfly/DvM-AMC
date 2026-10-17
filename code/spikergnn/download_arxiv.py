"""Download ogbn-arxiv and pre-set meta_info to skip prompts."""
import os, sys, urllib.request, zipfile, json, shutil
root = r'D:\spikergnn\DvM\data'
name = 'ogbn-arxiv'
os.makedirs(root, exist_ok=True)

# Meta info to skip update check
meta_dir = os.path.join(root, name); os.makedirs(meta_dir, exist_ok=True)
meta = {'version': '1', 'download_name': 'arxiv',
        'url': 'http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip',
        'add_inverse_edge': True, 'has_node_attr': True, 'num_tasks': 1,
        'additional_node_files': ['node_year'], 'additional_edge_files': [],
        'task_type': 'multiclass classification', 'num_classes': 40,
        'is_hetero': False, 'binary': False, 'eval_metric': 'acc'}
with open(os.path.join(meta_dir, 'meta_info.json'), 'w') as f:
    json.dump(meta, f, indent=2)
os.makedirs(os.path.join(meta_dir, 'raw'), exist_ok=True)

# Download
url = meta['url']
zip_path = os.path.join(meta_dir, 'raw', 'arxiv.zip')
print(f'Downloading {url} ...')
try:
    urllib.request.urlretrieve(url, zip_path)
    sz = os.path.getsize(zip_path)
    print(f'Downloaded: {sz/(1024*1024):.0f}MB')
except Exception as e:
    print(f'Download failed: {e}')
    sys.exit(1)

# Verify it's a valid zip
try:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print(f'Valid ZIP with {len(zf.namelist())} files')
except zipfile.BadZipFile:
    print('Corrupt ZIP — will need re-download')
    sys.exit(1)
print('Done!')
