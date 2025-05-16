import os, glob

def create_symlinks():
    """创建必要的符号链接"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    source_dir = os.path.join(project_root, 'docs/sphinx_doc/source')
    
    for md_file in glob.glob(os.path.join(project_root, '**/*.md'), recursive=True):
        if 'outputs' in md_file or 'sphinx_doc' in md_file:
            continue
            
        rel_path = os.path.relpath(md_file, project_root)
        target = os.path.join(source_dir, rel_path)
        
        os.makedirs(os.path.dirname(target), exist_ok=True)
        
        if not os.path.exists(target):
            rel_source = os.path.relpath(md_file, os.path.dirname(target))
            os.symlink(rel_source, target)
            print(f"Created symlink: {target} -> {rel_source}")
            
if __name__ == '__main__':
    create_symlinks()
