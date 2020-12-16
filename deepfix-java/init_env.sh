echo
echo 'Setting up a new virtual environment...'
echo
echo y | conda create -n deepfix python=2.7
echo 'done!'
source activate deepfix
pip install -i subprocess32 regex javac_parser

conda install tensorflow-gpu==1.0.1

echo 'Virtual environment configured.'

