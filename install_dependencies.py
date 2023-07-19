import subprocess

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

packages = ['pandas', 'nltk', 'scikit-learn', 'seaborn', 'matplotlib']

for package in packages:
    install(package)
