echo
echo 'Setting up a new virtual environment...'
echo
echo y | conda create -n rlassist python=2.7.15
echo
source activate rlassist
pip install --upgrade pip
pip install scipy==0.19.1 psutil subprocess32 regex cython unqlite javac_parser
pip install prettytable

pip install tensorflow-gpu==1.0.1

mkdir logs
mkdir data/network_inputs
mkdir data/checkpoints

echo 'Virtual environment configured.'

