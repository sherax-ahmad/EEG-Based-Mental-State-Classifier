from setuptools import setup, find_packages

setup(
    name="eeg-mental-state-classifier",
    version="0.1.0",
    description="EEG-based mental state classification: focus, stress, and sleep staging",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "mne>=1.4.0",
        "wfdb>=4.1.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.0.0", "jupyter>=1.0.0"],
        "wavelets": ["PyWavelets>=1.4.1"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
