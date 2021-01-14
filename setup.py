import glob
from setuptools import setup

setup_args = {
    "name": "ideotype",
    "author": "Jennifer Hsiao",
    "url": "https://github.com/jennhsiao/ideotype",
    "license": "MIT",
    "description": "maizsim ideotype project",
    "package_dir": {"ideotype": "ideotype"},
    "packages": ["ideotype"],
    "scripts": glob.glob("scripts/*"),
    "use_scm_version": True,
    "include_package_data": True,
    "install_requires": ["pandas", "pyyaml", "setuptools_scm"]
    # TODO: add more install_requires here
}

if __name__ == "__main__":
    setup(**setup_args)
