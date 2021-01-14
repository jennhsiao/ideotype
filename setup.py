import glob
from setuptools import setup

setup_args = {
    "name": "ideotype",
    "author": "Jennifer Hsiao",
    #"url": "https://github.com/RadioAstronomySoftwareGroup/pyradiosky",
    #"license": "MIT",
    #"description": "Python objects and interfaces for representing diffuse, extended and compact astrophysical radio sources",
    #"long_description": readme,
    #"long_description_content_type": "text/markdown",
    "package_dir": {"ideotype": "ideotype"},
    "packages": ["ideotype"],
    "scripts": glob.glob("scripts/*"),
    "use_scm_version": True,
    "include_package_data": True,
    "install_requires": ["pandas", "pyyaml", "setuptools_scm"]  # need to add more requires here
}

if __name__ == "__main__":
    setup(**setup_args)